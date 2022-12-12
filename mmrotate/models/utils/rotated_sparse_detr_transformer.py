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

try:
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention

except ImportError:
    warnings.warn(
        '`MultiScaleDeformableAttention` in MMCV has been moved to '
        '`mmcv.ops.multi_scale_deform_attn`, please update your MMCV')
    from mmcv.cnn.bricks.transformer import MultiScaleDeformableAttention

@ROTATED_TRANSFORMER.register_module()
class RotatedSparseDetrTransformer(Transformer):
    """Implements the Sparse DETR transformer.

    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
        use_enc_aux_loss (bool): Whether to use aux_loss at encoder.
            Default to False.
        rho (float): keeping ratio of top-p% query.
            Default to 0.
        eff_query_init (bool):
            Default to False.
        eff_specific_head (bool):
            Default to False.    
    """

    def __init__(self,
                 as_two_stage=False,
                 use_enc_aux_loss=False,
                 rho=0.,
                 eff_query_init=False,
                 eff_specific_head=False,
                 num_feature_levels=5,
                 two_stage_num_proposals=300,
                 **kwargs):
        super(RotatedSparseDetrTransformer, self).__init__(**kwargs)
        self.as_two_stage = as_two_stage
        self.use_enc_aux_loss = use_enc_aux_loss
        self.rho = rho

        self.eff_query_init = eff_query_init
        self.eff_specific_head = eff_specific_head

        self.num_feature_levels = num_feature_levels
        self.two_stage_num_proposals = two_stage_num_proposals
        self.embed_dims = self.encoder.embed_dims

        self.class_embed = None
        self.bbox_embed = None
        self.init_layers()


    def init_layers(self):
        """Initialize layers of the DeformableDetrTransformer."""
        if self.rho:
            self.enc_mask_predictor = MaskPredictor(self.embed_dims, self.embed_dims)
        else:
            self.enc_mask_predictor = None

        self.level_embeds = nn.Parameter(
            torch.Tensor(self.num_feature_levels, self.embed_dims))

        if self.as_two_stage:
            self.enc_output = nn.Linear(self.embed_dims, self.embed_dims)
            self.enc_output_norm = nn.LayerNorm(self.embed_dims)
            self.pos_trans = nn.Linear(self.embed_dims * 2,
                                       self.embed_dims * (1 if self.eff_query_init else 2))
            self.pos_trans_norm = nn.LayerNorm(self.embed_dims * (1 if self.eff_query_init else 2))
        else:
            self.reference_points = nn.Linear(self.embed_dims, 2)        

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MultiScaleDeformableAttention):
                m.init_weights()
        if not self.as_two_stage:
            xavier_init(self.reference_points, distribution='uniform', bias=0.)
        normal_(self.level_embeds)

    def gen_encoder_output_proposals(self, memory, memory_padding_mask,
                                     spatial_shapes):
        """Generate proposals from encoded memory.

        Args:
            memory (Tensor) : The output of encoder,
                has shape (bs, num_key, embed_dim).  num_key is
                equal the number of points on feature map from
                all level.
            memory_padding_mask (Tensor): Padding mask for memory.
                has shape (bs, num_key).
            spatial_shapes (Tensor): The shape of all feature maps.
                has shape (num_level, 2).

        Returns:
            tuple: A tuple of feature map and bbox prediction.

                - output_memory (Tensor): The input of decoder,  \
                    has shape (bs, num_key, embed_dim).  num_key is \
                    equal the number of points on feature map from \
                    all levels.
                - output_proposals (Tensor): The normalized proposal \
                    after a inverse sigmoid (only xywh), has shape \
                    (bs, num_keys, 5) format as  (x,y,w,h,a).
        """

        N, S, C = memory.shape
        proposals = []
        _cur = 0
        for lvl, (H, W) in enumerate(spatial_shapes):
            # level of encoded feature scale
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H * W)].view(
                N, H, W, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(
                torch.linspace(
                    0, H - 1, H, dtype=torch.float32, device=memory.device),
                torch.linspace(
                    0, W - 1, W, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1),
                               valid_H.unsqueeze(-1)], 1).view(N, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)
            angle = torch.zeros_like(mask_flatten_)
            proposal = torch.cat((grid, wh, angle), -1).view(N, -1, 5)
            proposals.append(proposal)
            _cur += (H * W)
        
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals[..., :4] > 0.01) & (output_proposals[..., :4] < 0.99)).all(-1, keepdim=True)

         # inverse of sigmoid
        output_proposals[..., :4] = torch.log(output_proposals[..., :4] / (1 - output_proposals[..., :4]))
        output_proposals[..., :4] = output_proposals[..., :4].masked_fill(
            memory_padding_mask.unsqueeze(-1), float('inf')) # sigmoid(inf) = 1
        output_proposals[..., :4] = output_proposals[..., :4].masked_fill(
            ~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid,
                                                  float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals, (~memory_padding_mask).sum(axis=-1)    

    def get_valid_ratio(self, mask):
        """Get the valid radios of feature maps of all  level."""
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def get_proposal_pos_embed(self,
                               proposals,
                               num_pos_feats=128,
                               temperature=10000):
        """Get the position embedding of proposal."""
        scale = 2 * math.pi
        dim_t = torch.arange(
            num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature**(2 * (dim_t // 2) / num_pos_feats)
        
        proposals = proposals.sigmoid() * scale # N, L, 4
        
        pos = proposals[:, :, :, None] / dim_t # N, L, 4, 128
        
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4)  # N, L, 4, 64, 2
        pos = pos.flatten(2)  # N, L, 512 (4 x 128)
        return pos

    def forward(self,
                mlvl_feats,
                mlvl_masks,
                query_embed,
                mlvl_pos_embeds,
                bbox_embed=None,
                class_embed=None,
                for_encoder=None,
                **kwargs):
        """Forward function for `Transformer`.

        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, embed_dims, h, w].
            mlvl_masks (list(Tensor)): The key_padding_mask from
                different level used for encoder and decoder,
                each element has shape  [bs, h, w].
            query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            mlvl_pos_embeds (list(Tensor)): The positional encoding
                of feats from different level, has the shape
                 [bs, embed_dims, h, w].
            bbox_embed (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. 
                Default to None.
            class_embed (obj:`nn.ModuleList`): Classification heads
                for feature maps from each decoder layer. Only would
                 be passed when `as_two_stage` is True. 
                 Default to None.
            for_encoder (encoder_class_embed, encoder_bbox_embed):
                Only would be passed when `use_enc_aux_loss` is True. 
                Default to None.
                - encoder_class_embed (obj:`nn.ModuleList`): 
                    Classification heads for feature maps from each encoder layer.
                - encoder_bbox_embed (obj:`nn.ModuleList`): 
                    Regression heads for feature maps from each encoder layer.

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - inter_states (Tensor): Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, two_stage_num_proposals, embed_dims), else has shape (bs, two_stage_num_proposals, embed_dims).
                - init_reference_out (Tensor): The initial value of reference \
                    points, has shape (bs, two_stage_num_proposals, 4).
                - inter_references_out (Tensor): The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs, two_stage_num_proposals, 4)
                - enc_outputs_class (Tensor): The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (bs, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact (Tensor): The regression results \
                    generated from encoder's feature maps, has shape \
                    (batch, h*w, 5). Only would \
                    be returned when `as_two_stage` is 'True', \
                    otherwise None.
                - backbone_mask_prediction (Tensor): The result of scoreing network with shape (bs, w*h). 
                    Only would be returned when `rho` is not 0, otherwise None.
                - enc_inter_outputs_class (List[Tensor]): Output class from all encoder layers' auxiliary head. 
                    Each Tensor has shape (bs, top-p%, num_class). 
                    Only would be returned when `encoder_aux_head` is 'True', otherwise None.
                - enc_inter_outputs_coord_unact (List[Tensor]): Output BBox from all encoder layers' auxiliary head. 
                    Each Tensor has shape (bs, top-p%, 5).
                    Only would be returned when `encoder_aux_head` is 'True', otherwise None.
                - sampling_locations_enc (Tensor): Sampling locations from all encoder layers.
                    Has shape (bs, num_layers, top-p%, num_heads, num_feat_levels, num_points, 2).
                - attn_weights_enc (Tensor): attention weights from all encoder layers.
                    Has shape (bs, num_layers, top-p%, num_heads, num_feat_levels, num_points).
                - sampling_locations_dec (Tensor): Sampling locations from all encoder layers.
                    Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points, 2).
                - attn_weights_dec (Tensor): attention weights from all encoder layers.
                    Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points).
                - backbone_topk_proposals (Tensor): Backbone query indexes in top-p% of w*h, has shape (bs, top-p%).
                - spatial_shapes (Tensor): Spatial_shapes of each feature level with shape (num_feat_levels, 2).
                - level_start_index (Tensor): Starting index of each feature level of src's dim=1 (w*h).
                    Has shape (num_feat_levels).
        """
        assert self.as_two_stage or query_embed is not None
        encoder_aux_head = False
        if for_encoder:
            encoder_aux_head = True
            encoder_class_embed, encoder_bbox_embed = for_encoder
        

        feat_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (feat, mask, pos_embed) in enumerate(zip(mlvl_feats, mlvl_masks, mlvl_pos_embeds)):
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)            
            feat = feat.flatten(2).transpose(1, 2) # [bs, w*h, c]            
            mask = mask.flatten(1) # [bs, w*h]
            pos_embed = pos_embed.flatten(2).transpose(1, 2) # [bs, w*h, c]            
            lvl_pos_embed = pos_embed + self.level_embeds[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            feat_flatten.append(feat)
            mask_flatten.append(mask)
        
        feat_flatten = torch.cat(feat_flatten, 1) # [bs, w*h, c]   
        mask_flatten = torch.cat(mask_flatten, 1) # [bs, w*h]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1) # [bs, w*h, c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=feat_flatten.device) # [num_feat_level=4, 2]
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # [num_feat_level]
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in mlvl_masks], 1) # [bs, num_feat_level, 2]

        ###########
        # prepare for sparse encoder
        if self.rho or self.use_enc_aux_loss:
            backbone_output_memory, backbone_output_proposals, valid_token_nums = \
                self.gen_encoder_output_proposals(
                    feat_flatten+lvl_pos_embed_flatten, mask_flatten, spatial_shapes)
            '''
            backbone_output_memory: [bs, w*h, c]
            backbone_output_proposals: [bs, w*h, 5]
            valid_token_nums: [w*h]
            '''

        
        if self.rho:
            sparse_token_nums = (valid_token_nums * self.rho).int() + 1
            backbone_topk = int(max(sparse_token_nums))
            self.sparse_token_nums = sparse_token_nums

            backbone_topk = min(backbone_topk, backbone_output_memory.shape[1])

            backbone_mask_prediction = self.enc_mask_predictor(backbone_output_memory).squeeze(-1) # [bs, w*h]
            # excluding pad area
            backbone_mask_prediction = backbone_mask_prediction.masked_fill(mask_flatten, backbone_mask_prediction.min())

            backbone_topk_proposals = torch.topk(backbone_mask_prediction, backbone_topk, dim=1)[1]  # [bs, top-p%]

        else:
            backbone_topk_proposals = None
            sparse_token_nums= None

        output_proposals = backbone_output_proposals.sigmoid() if self.use_enc_aux_loss else None  

        encoder_output = self.encoder(
            feat_flatten, 
            spatial_shapes, 
            level_start_index, 
            valid_ratios, 
            pos=lvl_pos_embed_flatten, 
            padding_mask=mask_flatten, 
            topk_inds=backbone_topk_proposals, 
            output_proposals=output_proposals,
            sparse_token_nums=sparse_token_nums,
            aux_heads = encoder_aux_head,
            class_embed=encoder_class_embed, 
            bbox_embed=encoder_bbox_embed,
            **kwargs)
        """
        encoder_output
            - memory (Tensor): Encoder output query with shape (bs, w*h, embed_dims).
            - sampling_locations_enc (Tensor): Sampling locations from all encoder layers.
                Has shape (bs, num_layers, top-p%, num_heads, num_feat_levels, num_points, 2).
            - attn_weights_enc (Tensor): attention weights from all encoder layers.
                Has shape (bs, num_layers, top-p%, num_heads, num_feat_levels, num_points).
            - enc_inter_outputs_class (List[Tensor]): If encoder_aux_head is False, None.
                Otherwise, output class from all encoder layers' auxiliary head. Each Tensor has shape (bs, top-p%, num_class).
            - enc_inter_outputs_coord_unact (List[Tensor]): If encoder_aux_head is False, None.
                Otherwise, output BBox from all encoder layers' auxiliary head. Each Tensor has shape (bs, top-p%, 5).
        """
        memory, sampling_locations_enc, attn_weights_enc = encoder_output[:3]
        
        if encoder_aux_head:
            enc_inter_outputs_class, enc_inter_outputs_coord_unact = encoder_output[3:5]
        

        ###########
        # prepare input for decoder
        bs, _, c = memory.shape
        if self.as_two_stage:
            # finalize the first stage output
            # project & normalize the memory and make proposal bounding boxes on them
            output_memory, output_proposal, _ = self.gen_encoder_output_proposals(
                    memory, mask_flatten, spatial_shapes) # output_proposals : [bs, w*h, 5]
            enc_output_proposal = inverse_sigmoid(enc_inter_outputs_coord_unact[-1])  # [bs, topk, 5]
            
            output_proposals = []
            for i in range(backbone_topk_proposals.shape[0]):
                output_proposals.append(output_proposal[i].scatter(0, backbone_topk_proposals[i][:sparse_token_nums[i]].unsqueeze(-1).repeat(1, enc_output_proposal.size(-1)), enc_output_proposal[i][:sparse_token_nums[i]]))
            output_proposals = torch.stack(output_proposals)


            # hack implementation for two-stage Deformable DETR (using the last layer registered in class/bbox_embed)

            # 1) a linear projection for bounding box binary classification (fore/background)
            enc_outputs_class = class_embed[self.decoder.num_layers](
                output_memory) # [bs, w*h, num_class]
            # 2) 3-layer FFN for bounding box regression
            enc_outputs_coord_offset= bbox_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = output_proposals + enc_outputs_coord_offset


            # top scoring bounding boxes are picked as the final region proposals. 
            # these proposals are fed into the decoder as initial boxes for the iterative bounding box refinement.
            topk = self.two_stage_num_proposals

            if self.eff_specific_head:
                # take the best score for judging objectness with class specific head
                enc_outputs_fg_class = enc_outputs_class.topk(1, dim=2).values[... , 0]

            else:
                # take the score from the binary(fore/background) classfier 
                # though outputs have 15 output dim, the 1st dim. alone will be used for the loss computation.
                enc_outputs_fg_class = enc_outputs_class[..., 0]

            topk_proposals = torch.topk(enc_outputs_fg_class, topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 5))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid() # [bs, 250, 5]


            init_reference_out = reference_points
            # pos_embed -> linear layer -> layer norm
            pos_trans_out = self.pos_trans_norm(
                self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact[..., :4])))

            
            if self.eff_query_init:
                # Efficient-DETR uses top-k memory as the initialization of `tgt` (query vectors)
                tgt = torch.gather(memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, memory.size(-1)))
                query_pos = pos_trans_out

            else:
                query_pos, tgt = torch.split(pos_trans_out, c, dim=2)
            
        else:
            query_pos, query = torch.split(query_embed, c, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            tgt = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos).sigmoid()
            init_reference_out = reference_points

        
        
        ###########
        # decoder
        inter_states, inter_references, sampling_locations_dec, attn_weights_dec = \
            self.decoder(tgt, 
                         reference_points, 
                         src=memory, 
                         src_spatial_shapes=spatial_shapes, 
                         src_level_start_index=level_start_index,
                         src_valid_ratios=valid_ratios, 
                         query_pos=query_pos, 
                         src_padding_mask=mask_flatten,
                         topk_inds=topk_proposals,
                         bbox_embed=bbox_embed)
        """
        inter_states (Tensor): Results with shape (num_layers, bs, two_stage_num_proposals, embed_dims).
        inter_references (Tensor): reference points with shape (num_layers, num_query, bs, embed_dims).
        sampling_locations_dec (Tensor): Sampling locations from all encoder layers.
            Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points, 2).
        attn_weights_dec (Tensor): attention weights from all encoder layers.
            Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points).
        """
 
        inter_references_out = inter_references

        ret = []
        ret += [inter_states, init_reference_out, inter_references_out]
        ret += [enc_outputs_class, enc_outputs_coord_unact] if self.as_two_stage else [None] * 2
        ret += [backbone_mask_prediction] if self.rho else [None]        
        ret += [enc_inter_outputs_class, enc_inter_outputs_coord_unact] if self.use_enc_aux_loss else [None] * 2
        ret += [sampling_locations_enc, attn_weights_enc, sampling_locations_dec, attn_weights_dec]
        ret += [backbone_topk_proposals, spatial_shapes, level_start_index]
        return ret

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RotatedSparseDetrTransformerEncoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.
    """

    def __init__(self, *args, **kwargs):

        super(RotatedSparseDetrTransformerEncoder, self).__init__(*args, **kwargs)
                
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

@TRANSFORMER_LAYER.register_module()
class RotatedSparseDetrTransformerEncoderLayer(BaseTransformerLayer):
    """Implements encoder layer in Sparse DETR transformer.
    """

    def __init__(self, *args, **kwargs):
        super(RotatedSparseDetrTransformerEncoderLayer, self).__init__(*args, **kwargs)

    def forward(self,
                src,
                tgt=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            src, tgt (Tensor): src is input flatten features with shape [bs, w*h, embed_dims].
                tgt is None or input sparsed flatten features with shape [bs, top-p%, embed_dims].
                If tgt is None: self-attention / if tgt is not None: cross-attention w.r.t. the target queries.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            src: forwarded results with shape [bs, top-p%, embed_dims].
            sampling_locations: [bs, top-p%, num_heads, num_feat_levels, num_points, 2].
            attn_weights: [bs, top-p%, num_heads, num_feat_levels, num_points].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = src
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                if tgt is None:
                    # self-attention
                    query = src
                else:
                    # cross-attention
                    query = tgt                  
                
                src, sampling_locations, attn_weights = self.attentions[attn_index](
                    query,
                    src,
                    src,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                """
                src: [bs, top-p%, embed_dims]
                sampling_locations: [bs, top-p%, num_heads, num_feat_levels, num_points, 2]
                attn_weights: [bs, top-p%, num_heads, num_feat_levels, num_points]
                """
                
                attn_index += 1
                identity = src

            elif layer == 'norm':
                src = self.norms[norm_index](src)
                norm_index += 1

            elif layer == 'ffn':
                src = self.ffns[ffn_index](
                    src, identity if self.pre_norm else None)
                ffn_index += 1

        return src, sampling_locations, attn_weights

@TRANSFORMER_LAYER_SEQUENCE.register_module()
class RotatedSparseDetrTransformerDecoder(TransformerLayerSequence):
    """Implements the decoder in Sparse DETR transformer.
    """

    def __init__(self, *args, return_intermediate=False, **kwargs):
        super(RotatedSparseDetrTransformerDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
            
    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, 
                src_valid_ratios, query_pos=None, src_padding_mask=None, topk_inds=None, bbox_embed=None):
        """Forward function for `TransformerDecoder`.

        Args:
            tgt (Tensor): Input query with shape
                `(bs, two_stage_num_proposals, embed_dims)`.
            reference_points (Tensor): The reference points of offset. 
                Has shape (bs, two_stage_num_proposals, 4) when as_two_stage,
                otherwise has shape (bs, two_stage_num_proposals, 2).
            src (Tensor): Last MS feature map from the encoder with shape (bs, w*h, embed_dims).
            src_spatial_shapes (Tensor): The shape of multi-scale features. 
                Has shape (num_feat_levels, 2). Dim=0 indicates feature level, dim=1 is (w,h).
            src_level_start_index (Tensor): Starting index of each feature level of src's dim=1 (w*h).
                Has shape (num_feat_levels).
            src_valid_ratios (Tensor): The radios of valid points on the feature map, has shape
                (bs, num_feat_levels, 2).
            query_pos (Tensor): Positional embedding, has shape (bs, w*h, embed_dims).
            src_padding_mask (Tensor): Indicates whether pixels are padded or not, has shape (bs, w*h)
            topk_inds (Tensor): query indexes in top-p% of w*h, has shape (bs, top-p%)
            bbox_embed: (obj:`nn.ModuleList`): Used for refining the regression results. 
                Only would be passed when with_box_refine is True,
                otherwise would be passed a `None`.

        Returns:
            output (Tensor): Results with shape (bs, two_stage_num_proposals, embed_dims) when
                return_intermediate is `False`, otherwise it has shape (num_layers, bs, two_stage_num_proposals, embed_dims).
            reference_points (Tensor): reference points with shape (bs, two_stage_num_proposals, 4)
                when return_intermediate is `False`, otherwise it has shape (num_layers, bs, two_stage_num_proposals, embed_dims).
            sampling_locations_all (Tensor): Sampling locations from all encoder layers.
                Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points, 2).
            attn_weights_all (Tensor): attention weights from all encoder layers.
                Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points).
        """        
        output = tgt

        intermediate = []
        intermediate_reference_points = []
        sampling_locations_all = []
        attn_weights_all = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == 5:
                """
                two stage or output from iterative bounding box refinement
                reference_points: bs, two_stage_num_proposals, 5(x/y/w/h/a)
                src_valid_ratios: bs, num_feature_levels, 2(w/h)
                reference_points_input: bs, two_stage_num_proposals, num_feature_levels, 5(x/y/w/h/a)
                """

                reference_points_input = reference_points[:, :, None, :4] * \
                                         torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else:
                assert reference_points.shape[-1] == 2, 'reference_point.shape[-1]={} is not available'.format(reference_points.shape[-1])
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]

            output, sampling_locations, attn_weights = \
                layer(query=output, 
                      key=None,
                      value=src,
                      query_pos=query_pos,                       
                      key_padding_mask=src_padding_mask,
                      reference_points=reference_points_input, 
                      spatial_shapes=src_spatial_shapes, 
                      level_start_index=src_level_start_index)                        
            """
            output (Tensor): forwarded results with shape [bs, two_stage_num_proposals, embed_dims].
            sampling_locations (Tensor): [bs, two_stage_num_proposals, num_heads, num_feat_levels, num_points, 2].
            attn_weights (Tensor): [bs, two_stage_num_proposals, num_heads, num_feat_levels, num_points].            
            """

            sampling_locations_all.append(sampling_locations)
            attn_weights_all.append(attn_weights)

            # implementation for iterative bounding box refinement
            if bbox_embed is not None:
                tmp = bbox_embed[lid](output)
                if reference_points.shape[-1] == 5:
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                else:                    
                    assert reference_points.shape[-1] == 2, 'reference_point.shape[-1]={} is not available'.format(reference_points.shape[-1])
                    new_reference_points = tmp
                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()

                reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(reference_points)
        
        sampling_locations_all = torch.stack(sampling_locations_all, dim=1)
        attn_weights_all = torch.stack(attn_weights_all, dim=1)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), sampling_locations_all, attn_weights_all

        return output, reference_points, sampling_locations_all, attn_weights_all

@TRANSFORMER_LAYER.register_module()
class RotatedSparseDetrTransformerDecoderLayer(BaseTransformerLayer):
    """Implements decoder layer in Sparse DETR transformer.

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(RotatedSparseDetrTransformerDecoderLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        assert len(operation_order) == 6
        assert set(operation_order) == set(
            ['self_attn', 'norm', 'cross_attn', 'ffn'])

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape [bs, two_stage_num_proposals, embed_dims].
            key (Tensor): The key tensor with shape [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape [bs, w*h, embed_dims].
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            query (Tensor): forwarded results with shape [bs, two_stage_num_proposals, embed_dims].
            sampling_locations (Tensor): [bs, two_stage_num_proposals, num_heads, num_feat_levels, num_points, 2].
            attn_weights (Tensor): [bs, two_stage_num_proposals, num_heads, num_feat_levels, num_points].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query

                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':

                query, sampling_locations, attn_weights = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                identity = query
  
            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)
                ffn_index += 1

        return query, sampling_locations, attn_weights

class MaskPredictor(nn.Module):
    '''
    Scoring Network of Sparse DETR
    '''
    def __init__(self, in_dim, h_dim):
        super().__init__()
        self.h_dim = h_dim
        self.layer1 = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, h_dim),
            nn.GELU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h_dim, h_dim // 2),
            nn.GELU(),
            nn.Linear(h_dim // 2, h_dim // 4),
            nn.GELU(),
            nn.Linear(h_dim // 4, 1)
        )
    
    def forward(self, x):
        z = self.layer1(x)
        z_local, z_global = torch.split(z, self.h_dim // 2, dim=-1)
        z_global = z_global.mean(dim=1, keepdim=True).expand(-1, z_local.shape[1], -1)
        z = torch.cat([z_local, z_global], dim=-1)
        out = self.layer2(z)
        return out

@ATTENTION.register_module()
class SparseDETRAttention(MultiScaleDeformableAttention):
    """An attention module used in Sparse-Detr.
    """
    def __init__(self, *args, **kwargs):
        super(SparseDETRAttention, self).__init__(*args, **kwargs)
        self.batch_first=True

    @deprecated_api_warning({'residual': 'identity'},
                            cls_name='SparseDETRAttention')
    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (torch.Tensor): The positional encoding for `key`. Default
                None.
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """
        
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets \
                / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points \
                * reference_points[:, :, None, :, None, 2:] \
                * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        if torch.cuda.is_available() and value.is_cuda:
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity, sampling_locations, attention_weights



