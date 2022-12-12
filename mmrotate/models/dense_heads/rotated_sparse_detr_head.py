# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import force_fp32

from mmdet.core import multi_apply
from mmdet.models.utils.transformer import inverse_sigmoid
from ..builder import ROTATED_HEADS
from .rotated_detr_head import RotatedDETRHead


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

@ROTATED_HEADS.register_module()
class RotatedSparseDETRHead(RotatedDETRHead):
    """Head of Rotated Sparse DETR

    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder. Default to False.
        use_enc_aux_loss (bool): Whether to use aux_loss at encoder.
            Default to False.
        rho (float): keeping ratio of top-k% query.
            Default to 0.

        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
    """

    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 use_enc_aux_loss=False,
                 rho=0.,
                 transformer=None,
                 random_refpoints_xy=False,
                 loss_scoreing_network_weight=1.0,
                 **kwargs):
        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        self.use_enc_aux_loss = use_enc_aux_loss
        self.rho = rho

        self.random_refpoints_xy = random_refpoints_xy
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if self.use_enc_aux_loss:
            transformer['use_enc_aux_loss'] = self.use_enc_aux_loss
        if self.rho:
            transformer['rho'] = self.rho
        self.num_refine_stages = 2

        self.loss_scoreing_network_weight = loss_scoreing_network_weight

        super(RotatedSparseDETRHead, self).__init__(
            *args, transformer=transformer, **kwargs)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""

        self.class_embed = nn.Linear(self.embed_dims, self.cls_out_channels)
        self.bbox_embed = MLP(self.embed_dims, self.embed_dims, output_dim=5, num_layers=3)

        if self.use_enc_aux_loss:
            self.encoder_class_embed = nn.Linear(self.embed_dims, self.cls_out_channels)
            self.encoder_bbox_embed = MLP(self.embed_dims, self.embed_dims, output_dim=5, num_layers=3)        

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])     
        
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.cls_out_channels) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)

        if self.use_enc_aux_loss:
            self.encoder_class_embed.bias.data = torch.ones(self.cls_out_channels) * bias_value
            nn.init.constant_(self.encoder_bbox_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self.encoder_bbox_embed.layers[-1].bias.data, 0)

        # at each layer of decoder (by default)
        num_pred = self.transformer.decoder.num_layers
        if self.as_two_stage:
            # last reg_branch is used to generate proposal from encode feature map when as_two_stage is True.
            num_pred += 1  
        if self.use_enc_aux_loss:
            # at each layer of encoder (excl. the last)
            num_pred += self.transformer.encoder.num_layers - 1 
                
        if self.with_box_refine or self.use_enc_aux_loss:
            # individual heads with the same initializatio
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)

            if self.use_enc_aux_loss:
                self.encoder_class_embed = _get_clones(self.encoder_class_embed, num_pred)
                self.encoder_bbox_embed = _get_clones(self.encoder_bbox_embed, num_pred)
                nn.init.constant_(self.encoder_bbox_embed[0].layers[-1].bias.data[2:], -2.0)
        else:
            # shared heads
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])

            if self.use_enc_aux_loss:
                nn.init.constant_(self.encoder_bbox_embed.layers[-1].bias.data[2:], -2.0)
                self.encoder_class_embed = nn.ModuleList([self.encoder_class_embed for _ in range(num_pred)])
                self.encoder_bbox_embed = nn.ModuleList([self.encoder_bbox_embed for _ in range(num_pred)])
        
        if self.as_two_stage:
            self.decoder_class_embed = self.class_embed
            self.decoder_bbox_embed = self.bbox_embed            
            for box_embed in self.decoder_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)         
        else:
            self.query_embedding = nn.Embedding(self.num_query, self.embed_dims * 2)
        
        if self.use_enc_aux_loss:
            # the output from the last layer should be specially treated as an input of decoder
            num_layers_excluding_the_last = self.transformer.encoder.num_layers - 1
            self.encoder_class_embed = self.encoder_class_embed[-num_layers_excluding_the_last:]
            self.encoder_bbox_embed = self.encoder_bbox_embed[-num_layers_excluding_the_last:] 

            for box_embed in self.encoder_bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
 
    def forward(self, mlvl_feats, img_metas):
        """Forward function.

        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 4D-tensor with shape
                (N, C, H, W).
            img_metas (list[dict]): List of image information.

        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, h, a). \
                Shape [nb_dec, bs, num_query, 5].
            enc_outputs_class (Tensor): The score of each point on encode \
                feature map, has shape (N, h*w, num_class). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
            enc_outputs_coord (Tensor): The proposal generate from the \
                encode feature map, has shape (N, h*w, 5). Only when \
                as_two_stage is True it would be returned, otherwise \
                `None` would be returned.
        """


        batch_size = mlvl_feats[0].size(0)
        input_img_h, input_img_w = img_metas[0]['batch_input_shape']
        img_masks = mlvl_feats[0].new_ones(
            (batch_size, input_img_h, input_img_w))
        for img_id in range(batch_size):
            img_h, img_w, _ = img_metas[img_id]['img_shape']
            img_masks[img_id, :img_h, :img_w] = 0

        mlvl_masks = []
        mlvl_positional_encodings = []
        spatial_shapes = []
        for feat in mlvl_feats:
            bs, c, h, w = feat.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            mlvl_masks.append(
                F.interpolate(img_masks[None],
                              size=feat.shape[-2:]).to(torch.bool).squeeze(0))
            mlvl_positional_encodings.append(
                self.positional_encoding(mlvl_masks[-1]))

        ###########
        # Transformer encoder & decoder
        if self.as_two_stage:
            query_embeds = None
        else:
            query_embeds = self.query_embedding.weight
        
                   
        (hs, init_reference, inter_references, enc_outputs_class, enc_outputs_coord,
            backbone_mask_prediction, enc_inter_outputs_class, enc_inter_outputs_coord, 
            sampling_locations_enc, attn_weights_enc, sampling_locations_dec, 
            attn_weights_dec, backbone_topk_proposals, spatial_shapes, level_start_index) = \
            self.transformer(
            mlvl_feats,
            mlvl_masks,
            query_embeds,
            mlvl_positional_encodings,
            bbox_embed=self.decoder_bbox_embed if self.with_box_refine else None,
            class_embed=self.decoder_class_embed if self.as_two_stage else None, 
            for_encoder=(self.encoder_class_embed,
                            self.encoder_bbox_embed) if self.use_enc_aux_loss else None,
        )
        """
        - hs (Tensor): Outputs from decoder. 
            If return_intermediate_dec is 'True' output has shape (num_dec_layers, bs, two_stage_num_proposals, embed_dims), else has shape (bs, two_stage_num_proposals, embed_dims).
        - init_reference (Tensor): The initial value of reference points, 
            has shape (bs, two_stage_num_proposals, 4).
        - inter_references (Tensor): The internal value of reference points in decoder, 
            has shape (num_dec_layers, bs, two_stage_num_proposals, 4).
        - enc_outputs_class (Tensor): The classification score of proposals generated from 
            encoder's feature maps, has shape (bs, h*w, num_classes).
            Only would be returned when `as_two_stage` is 'True', otherwise 'None'.
        - enc_outputs_coord (Tensor): The regression results generated from encoder's feature maps, 
            has shape (batch, h*w, 5). 
            Only would be returned when `as_two_stage` is 'True', otherwise 'None'.
        - backbone_mask_prediction (Tensor): The result of scoreing network with shape (bs, w*h). 
            Only would be returned when `rho` is not 0, otherwise 'None'.
        - enc_inter_outputs_class (List[Tensor]): Output class from all encoder layers' auxiliary head. 
            Each Tensor has shape (bs, top-p%, num_class). 
            Only would be returned when `encoder_aux_head` is 'True', otherwise 'None'.
        - enc_inter_outputs_coord (List[Tensor]): Output BBox from all encoder layers' auxiliary head. 
            Each Tensor has shape (bs, top-p%, 5).
            Only would be returned when `encoder_aux_head` is 'True', otherwise 'None'.
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
        
        ###########
        # Detection heads
        outputs_classes = []
        outputs_coords = []

        for lvl in range(hs.shape[0]):
            # lvl: level of decoding layer
            outputs_class = self.class_embed[lvl](hs[lvl])
            outputs_coord = self.bbox_embed[lvl](hs[lvl])

            assert init_reference is not None and inter_references is not None
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            reference = inverse_sigmoid(reference)            

            if reference.shape[-1] == 5:
                outputs_coord += reference
            else:
                assert reference.shape[-1] == 2
                outputs_coord[..., :2] += reference
            
            outputs_coord = outputs_coord.sigmoid()

            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        
        outputs_classes = torch.stack(outputs_classes)
        outputs_coords = torch.stack(outputs_coords)
        
        ret = [outputs_classes, outputs_coords]
        ret += [enc_outputs_class, enc_outputs_coord.sigmoid()] if self.as_two_stage else [None] *2
        ret += [backbone_mask_prediction] if self.rho else [None]
        ret += [enc_inter_outputs_class, enc_inter_outputs_coord] if self.use_enc_aux_loss else [None] * 2        
        ret += [sampling_locations_dec, attn_weights_dec]
        ret += [spatial_shapes, level_start_index]
        ret += [self.transformer.sparse_token_nums] if self.rho else [None] # sparse_token_nums
        ret += [torch.cat([m.flatten(1) for m in mlvl_masks], 1)] # mask_flatten
        
        return ret

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def loss(self,
             all_cls_scores,
             all_bbox_preds,
             enc_cls_scores,
             enc_bbox_preds, 
             backbone_mask_prediction, # loss of scoring network     
             enc_inter_cls_scores, # enc aux loss
             enc_inter_bbox_preds, # enc aux loss
            #  sampling_locations_enc,  
            #  attn_weights_enc, 
             sampling_locations_dec, # loss of scoring network 
             attn_weights_dec, # loss of scoring network 
            #  backbone_topk_proposals, 
             spatial_shapes, # loss of scoring network 
             level_start_index, # loss of scoring network 
             sparse_token_nums, # loss of scoring network 
             mask_flatten, # loss of scoring network 
             gt_bboxes_list,
             gt_labels_list,
             img_metas,
             gt_bboxes_ignore=None):
        """"Loss function.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h, a) and shape
                [nb_dec, bs, num_query, 5].
            enc_cls_scores (Tensor): Classification scores of
                points on encode feature map , has shape
                (N, h*w, num_classes). Only be passed when as_two_stage is
                True, otherwise is None.
            enc_bbox_preds (Tensor): Regression results of each points
                on the encode feature map, has shape (N, h*w, 5). Only be
                passed when as_two_stage is True, otherwise is None.
            sampling_locations_dec (Tensor): Sampling locations from all encoder layers.
                Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points, 2).
            attn_weights_dec (Tensor): attention weights from all encoder layers.
                Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points).
            spatial_shapes (Tensor): Spatial_shapes of each feature level with shape (num_feat_levels, 2).
            level_start_index (Tensor): Starting index of each feature level of src's dim=1 (w*h).
                Has shape (num_feat_levels).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 5) in [x,y,w,h,a] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            img_metas (list[dict]): List of image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        num_dec_layers = len(all_cls_scores)
        all_gt_rbboxes_list = [gt_bboxes_list for _ in range(num_dec_layers)]
        # all_gt_rbboxes_list = [gt_masks for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]
        all_gt_bboxes_ignore_list = [
            gt_bboxes_ignore for _ in range(num_dec_layers)
        ]
        img_metas_list = [img_metas for _ in range(num_dec_layers)]

        losses_cls, losses_bbox, losses_piou = multi_apply(
            self.loss_single, all_cls_scores, all_bbox_preds,
            all_gt_rbboxes_list, all_gt_labels_list, img_metas_list,
            all_gt_bboxes_ignore_list)

        loss_dict = dict()
        # loss of proposal generated from encode feature map.
        if enc_cls_scores is not None:
            binary_labels_list = [
                torch.zeros_like(gt_labels_list[i])
                for i in range(len(img_metas))
            ]
            enc_loss_cls, enc_losses_bbox, enc_losses_piou = \
                self.loss_single(enc_cls_scores, enc_bbox_preds,
                                 gt_bboxes_list, binary_labels_list,
                                 img_metas, gt_bboxes_ignore)
            loss_dict['enc_loss_cls'] = enc_loss_cls
            loss_dict['enc_loss_bbox'] = enc_losses_bbox
            loss_dict['enc_loss_piou'] = enc_losses_piou

        # aux loss of encoder
        if enc_inter_cls_scores is not None:
            num_enc_layers = len(enc_inter_cls_scores)
            all_gt_rbboxes_list = [gt_bboxes_list for _ in range(num_enc_layers)]
            # all_gt_rbboxes_list = [gt_masks for _ in range(num_dec_layers)]
            all_gt_labels_list = [gt_labels_list for _ in range(num_enc_layers)]
            all_gt_bboxes_ignore_list = [
                gt_bboxes_ignore for _ in range(num_enc_layers)
            ]
            img_metas_list = [img_metas for _ in range(num_enc_layers)]

            enc_inter_loss_cls, enc_inter_loss_bbox, enc_inter_loss_piou = multi_apply(
                self.loss_single, enc_inter_cls_scores, enc_inter_bbox_preds,
                all_gt_rbboxes_list, all_gt_labels_list, img_metas_list,
                all_gt_bboxes_ignore_list)
            
            # loss from encoder layers
            num_enc_layer = 0
            for loss_cls_i, loss_bbox_i, loss_iou_i in zip(enc_inter_loss_cls,
                                                       enc_inter_loss_bbox,
                                                       enc_inter_loss_piou):
                loss_dict[f'd{num_enc_layer}.enc_loss_cls'] = loss_cls_i
                loss_dict[f'd{num_enc_layer}.enc_loss_bbox'] = loss_iou_i
                loss_dict[f'd{num_enc_layer}.enc_loss_piou'] = loss_bbox_i
                num_enc_layer += 1

        
        # loss of scoring network 
        if backbone_mask_prediction is not None:
            flat_grid_attn_map_dec = self.attn_map_to_flat_grid(spatial_shapes, level_start_index, sampling_locations_dec, attn_weights_dec).sum(dim=(1,2)) # (bs, w*h)

            if mask_flatten is not None:
                flat_grid_attn_map_dec = flat_grid_attn_map_dec.masked_fill(
                mask_flatten, flat_grid_attn_map_dec.min()-1)

            num_topk = sparse_token_nums.max()
            topk_idx_tgt = torch.topk(flat_grid_attn_map_dec, num_topk)[1]
            target = torch.zeros_like(backbone_mask_prediction)
            for i in range(target.shape[0]):
                target[i].scatter_(0, topk_idx_tgt[i][:sparse_token_nums[i]], 1)
            
            loss_scoreing_network = F.multilabel_soft_margin_loss(backbone_mask_prediction, target)
            loss_dict['loss_scoreing_network'] = loss_scoreing_network * self.loss_scoreing_network_weight
            
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_piou'] = losses_piou[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_cls_i, loss_bbox_i, loss_iou_i in zip(losses_cls[:-1],
                                                       losses_bbox[:-1],
                                                       losses_piou[:-1]):
            loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            loss_dict[f'd{num_dec_layer}.loss_piou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i
            num_dec_layer += 1
        return loss_dict

    def attn_map_to_flat_grid(self, spatial_shapes, level_start_index, sampling_locations, attention_weights):
        """
        Args:
            sampling_locations_dec (Tensor): Sampling locations from all encoder layers.
                Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points, 2).
            attn_weights_dec (Tensor): attention weights from all encoder layers.
                Has shape (bs, num_layers, two_stage_num_proposals, num_heads, num_feat_levels, num_points).
            spatial_shapes (Tensor): Spatial_shapes of each feature level with shape (num_feat_levels, 2).
            level_start_index (Tensor): Starting index of each feature level of src's dim=1 (w*h).
                Has shape (num_feat_levels).

        Return:
            (Tensor) : (bs, num_layers, num_heads, w*h)
        """
        N, n_layers, _, n_heads, *_ = sampling_locations.shape
        sampling_locations = sampling_locations.permute(0, 1, 3, 2, 5, 4, 6).flatten(0, 2).flatten(1, 2) # [bs * num_layers * num_heads, two_stage_num_proposals * num_points, num_feat_levels, 2]
        attention_weights = attention_weights.permute(0, 1, 3, 2, 5, 4).flatten(0, 2).flatten(1, 2) # [bs * num_layers * num_heads, two_stage_num_proposals * num_points, num_feat_levels]

        rev_spatial_shapes = torch.stack([spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1) # hw -> wh (xy)
        col_row_float = sampling_locations * rev_spatial_shapes # [bs * num_layers * num_heads, two_stage_num_proposals * num_points, num_feat_levels, 2]        

        col_row_ll = col_row_float.floor().to(torch.int64)
        zero = torch.zeros(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
        one = torch.ones(*col_row_ll.shape[:-1], dtype=torch.int64, device=col_row_ll.device)
        col_row_lh = col_row_ll + torch.stack([zero, one], dim=-1)
        col_row_hl = col_row_ll + torch.stack([one, zero], dim=-1)
        col_row_hh = col_row_ll + 1

        margin_ll = (col_row_float - col_row_ll).prod(dim=-1)
        margin_lh = -(col_row_float - col_row_lh).prod(dim=-1)
        margin_hl = -(col_row_float - col_row_hl).prod(dim=-1)
        margin_hh = (col_row_float - col_row_hh).prod(dim=-1)

        flat_grid_shape = (attention_weights.shape[0], int(torch.sum(spatial_shapes[..., 0] * spatial_shapes[..., 1])))
        flat_grid = torch.zeros(flat_grid_shape, dtype=torch.float32, device=attention_weights.device)

        zipped = [(col_row_ll, margin_hh), (col_row_lh, margin_hl), (col_row_hl, margin_lh), (col_row_hh, margin_ll)]
        for col_row, margin in zipped:
            valid_mask = torch.logical_and(
                torch.logical_and(col_row[..., 0] >= 0, col_row[..., 0] < rev_spatial_shapes[..., 0]),
                torch.logical_and(col_row[..., 1] >= 0, col_row[..., 1] < rev_spatial_shapes[..., 1]),
            )
            idx = col_row[..., 1] * spatial_shapes[..., 1] + col_row[..., 0] + level_start_index
            idx = (idx * valid_mask).flatten(1, 2)
            weights = (attention_weights * valid_mask * margin).flatten(1)
            flat_grid.scatter_add_(1, idx, weights)
        
        return flat_grid.reshape(N, n_layers, n_heads, -1)

    @force_fp32(apply_to=('all_cls_scores_list', 'all_bbox_preds_list'))
    def get_bboxes(self,
                   all_cls_scores,
                   all_bbox_preds,
                   img_metas,
                   rescale=False):
        """Transform network outputs for a batch into bbox predictions.

        Args:
            all_cls_scores (Tensor): Classification score of all
                decoder layers, has shape
                [nb_dec, bs, num_query, cls_out_channels].
            all_bbox_preds (Tensor): Sigmoid regression
                outputs of all decode layers. Each is a 4D-tensor with
                normalized coordinate format (cx, cy, w, h, a) and shape
                [nb_dec, bs, num_query, 5].
            img_metas (list[dict]): Meta information of each image.
            rescale (bool, optional): If True, return boxes in original
                image space. Default False.

        Returns:
            list[list[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 6) tensor, where the first 4 columns \
                are bounding box positions (x,y,w,h,a) and the \
                6-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        cls_scores = all_cls_scores[-1]
        bbox_preds = all_bbox_preds[-1]

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score = cls_scores[img_id]
            bbox_pred = bbox_preds[img_id]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(cls_score, bbox_pred,
                                                img_shape, scale_factor,
                                                rescale)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_score,
                           bbox_pred,
                           img_shape,
                           scale_factor,
                           rescale=False):
        """Transform outputs from the last decoder layer into bbox predictions
        for each image.

        Args:
            cls_score (Tensor): Box score logits from the last decoder layer
                for each image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from the last decoder layer
                for each image, with coordinate format (cx, cy, w, h, a) and
                shape [num_query, 5].
            img_shape (tuple[int]): Shape of input image, (height, width, 3).
            scale_factor (ndarray, optional): Scale factor of the image arange
                as (w_scale, h_scale, w_scale, h_scale).
            rescale (bool, optional): If True, return boxes in original image
                space. Default False.

        Returns:
            tuple[Tensor]: Results of detected bboxes and labels.

                - det_bboxes: Predicted bboxes with shape [num_query, 6], \
                    where the first 5 columns are bounding box positions \
                    (x,y,w,h,a) and the 6-th column are scores \
                    between 0 and 1.
                - det_labels: Predicted labels of the corresponding box with \
                    shape [num_query].
        """
        assert len(cls_score) == len(bbox_pred)
        max_per_img = self.test_cfg.get('max_per_img', self.num_query)
        # exclude background
        if self.loss_cls.use_sigmoid:
            cls_score = cls_score.sigmoid()
            scores, indexes = cls_score.view(-1).topk(max_per_img)
            det_labels = indexes % self.num_classes
            bbox_index = indexes // self.num_classes
            bbox_pred = bbox_pred[bbox_index]
        else:
            scores, det_labels = F.softmax(cls_score, dim=-1)[..., :-1].max(-1)
            scores, bbox_index = scores.topk(max_per_img)
            bbox_pred = bbox_pred[bbox_index]
            det_labels = det_labels[bbox_index]

        bbox_pred[..., :4] = bbox_pred[..., :4] * img_shape[1]
        if rescale:
            bbox_pred[:, :4] /= bbox_pred[:, :4].new_tensor(scale_factor)
        det_bboxes = torch.cat((bbox_pred, scores.unsqueeze(1)), -1)

        return det_bboxes, det_labels

    def simple_test_bboxes(self, feats, img_metas, rescale=False):
        """Test det bboxes without test-time augmentation.

        Args:
            feats (tuple[torch.Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 6),
                where 5 represent (x,y,w,h,a, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,)
        """
        # forward of this head requires img_metas
        outs = self.forward(feats, img_metas)
        all_cls_scores, all_bbox_preds = outs[:2]
        results_list = self.get_bboxes(all_cls_scores, all_bbox_preds, img_metas, rescale=rescale)
        return results_list