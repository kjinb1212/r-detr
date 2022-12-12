import copy
import math
import warnings

import torch
import torch.nn as nn
from mmcv import deprecated_api_warning
from mmcv.cnn import xavier_init
from mmcv.cnn.bricks.registry import (TRANSFORMER_LAYER, TRANSFORMER_LAYER_SEQUENCE, ATTENTION)
from mmcv.cnn.bricks.transformer import (BaseTransformerLayer,
                                         TransformerLayerSequence)
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch

from torch.nn.init import normal_

from .builder import ROTATED_TRANSFORMER
from mmdet.models.utils import Transformer
from mmdet.models.utils.transformer import inverse_sigmoid
from mmrotate.ops.token_ops import TokenLearner, TokenFuser
try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention
    
@TRANSFORMER_LAYER_SEQUENCE.register_module()
class TokenLearningTransformerEncoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    """

    def __init__(self, *args, out_token=8, insert_learner=3, **kwargs):
        super(TokenLearningTransformerEncoder, self).__init__(*args, **kwargs)
        learners = []
        fusers = []
        for i in range(4):
            learners.append(TokenLearner(out_token=out_token* 2**i, emb=self.embed_dims))
            fusers.append(TokenFuser(in_token=out_token* 2**i, emb=self.embed_dims))
        self.learners = nn.ModuleList(learners)
        self.fusers = nn.ModuleList(fusers)
        self.insert_learner = insert_learner
        
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """Make reference points for every single point on the multi-scale feature maps.
        Each point has K reference points on every the multi-scale features.

        Args:
            spatial_shapes (Tensor): The shape of all
                feature maps, has shape (num_level, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            device (obj:`device`): The device where
                reference_points should be.

        Returns:
            Tensor: reference points used in decoder, has \
                shape (bs, num_keys, num_levels, 2).
        """
        reference_points_list = []
        for lvl, (H, W) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=torch.float32, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 1] * H)
            ref_x = ref_x.reshape(-1)[None] / (
                    valid_ratios[:, None, lvl, 0] * W)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    
    def forward(self,
                src,                
                spatial_shapes,
                level_start_index,
                valid_ratios,
                pos=None,
                padding_mask=None,
                topk_inds=None,
                output_proposals=None, 
                sparse_token_nums=None,
                aux_heads = False,
                class_embed=None, 
                bbox_embed=None,
                **kwargs):
        """Forward function for 'Sparse DETR Transformer Encoder'.

        Args:
            src (Tensor): Input query with shape (bs, w*h, embed_dims)`.
            spatial_shapes (Tensor): The shape of multi-scale features. Has shape (num_feat_levels, 2).
                Dim=0 indicates feature level, dim=1 is (w,h).
            level_start_index (Tensor): Starting index of each feature level of src's dim=1 (w*h).
                Has shape (num_feat_levels).
            valid_ratios (Tensor): The radios of valid points on the feature map, has shape
                (bs, num_feat_levels, 2).
            pos (Tensor): Positional embedding, has shape (bs, w*h, embed_dims).
            padding_mask (Tensor): Indicates whether pixels are padded or not, has shape (bs, w*h)
            topk_inds (Tensor): query indexes in top-p% of w*h, has shape (bs, top-p%)
            output_proposals (Tensor): backbone output proposals, has shape (bs, w*h, 5)
            sparse_token_nums (Tensor): length of top-p% query for each batch, has shape (bs).
            aux_heads (bool): Whether to use aux_head or not.
                Default False.
            class_embed: (obj:`nn.ModuleList`): Classification heads for feature maps from each encoder layer.
            bbox_embed: (obj:`nn.ModuleList`): BBox regression heads for feature maps from each encoder layer.

        Returns:
            output (Tensor): Encoder output query with shape (bs, w*h, embed_dims).
            sampling_locations_all (Tensor): Sampling locations from all encoder layers.
                Has shape (bs, num_layers, top-p%, num_heads, num_feat_levels, num_points, 2).
            attn_weights_all (Tensor): attention weights from all encoder layers.
                Has shape (bs, num_layers, top-p%, num_heads, num_feat_levels, num_points).

            If aux_heads is True, also return enc_inter_outputs_class, enc_inter_outputs_coords.
                enc_inter_outputs_class (List[Tensor]): Output class from all encoder layers' auxiliary head. Each Tensor has shape (bs, top-p%, num_class).
                enc_inter_outputs_coords (List[Tensor]): Output BBox from all encoder layers' auxiliary head. Each Tensor has shape (bs, top-p%, 5).
        """

        output = src
        sparsified_keys = False if topk_inds is None else True
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device) # (bs, w*h, num_feat_levels, 2)

        sampling_locations_all = []
        attn_weights_all = []
        if aux_heads:
            enc_inter_outputs_class = []
            enc_inter_outputs_coords = []
        if sparsified_keys:
            assert topk_inds is not None
            B_, N_, S_, P_ = reference_points.shape
            reference_points = torch.gather(reference_points.view(B_, N_, -1), 1, topk_inds.unsqueeze(-1).repeat(1, 1, S_*P_)).view(B_, -1, S_, P_) # [bs, top-p%, num_feat_levels, 2]

            tgt = torch.gather(output, 1, topk_inds.unsqueeze(-1).repeat(1, 1, output.size(-1))) # [bs, top-p%, embed_dims]
            pos = torch.gather(pos, 1, topk_inds.unsqueeze(-1).repeat(1, 1, pos.size(-1))) # [bs, top-p%, embed_dims]
            if output_proposals is not None:
                output_proposals = output_proposals.gather(1, topk_inds.unsqueeze(-1).repeat(1, 1, output_proposals.size(-1))) # [bs, top-p%, 5]
        else:
            tgt = None

        for lid, layer in enumerate(self.layers):
            if lid >= self.insert_learner:
                shortcut = output
                st_idx = 0
                output_list = []
                for lvl, (H, W) in enumerate(spatial_shapes):
                    output_list.append(self.learners[lvl](output[:, st_idx  : H*W], H, W))
                output = torch.cat(output_list, 1)
                tgt, sampling_locations, attn_weights = layer(
                output,
                tgt=output,
                query_pos=None,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                query_key_padding_mask=padding_mask,
                **kwargs)
                print('TokenLearningTransformerEncoder.forward() 하다 말았음')
                exit()
                
            
            
            # if tgt is None: self-attention / if tgt is not None: cross-attention w.r.t. the target queries
            tgt, sampling_locations, attn_weights = layer(
                output,
                tgt=tgt,
                query_pos=pos,
                reference_points=reference_points,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                query_key_padding_mask=padding_mask,
                **kwargs)
            
            sampling_locations_all.append(sampling_locations)
            attn_weights_all.append(attn_weights)

            if sparsified_keys:                
                if sparse_token_nums is None:
                    output = output.scatter(1, topk_inds.unsqueeze(-1).repeat(1, 1, tgt.size(-1)), tgt)
                else:
                    outputs = []
                    for i in range(topk_inds.shape[0]):
                        outputs.append(output[i].scatter(0, topk_inds[i][:sparse_token_nums[i]].unsqueeze(-1).repeat(1, tgt.size(-1)), tgt[i][:sparse_token_nums[i]]))
                    output = torch.stack(outputs)
            else:
                output = tgt
            
            if aux_heads and lid < self.num_layers - 1:
                # feed outputs to aux. heads
                output_class = class_embed[lid](tgt)
                output_offset = bbox_embed[lid](tgt)
                output_proposals = inverse_sigmoid(output_proposals) + output_offset
                output_proposals = output_proposals.sigmoid().detach()

                # values to be used for loss compuation
                enc_inter_outputs_class.append(output_class)
                enc_inter_outputs_coords.append(output_proposals)

        sampling_locations_all = torch.stack(sampling_locations_all, dim=1)
        attn_weights_all = torch.stack(attn_weights_all, dim=1)
 
        if aux_heads:
            return output, sampling_locations_all, attn_weights_all, enc_inter_outputs_class, enc_inter_outputs_coords
        return output, sampling_locations_all, attn_weights_all