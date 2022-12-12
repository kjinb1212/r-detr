import warnings

import torch
import numpy as np

from ..builder import ROTATED_DETECTORS
from .single_stage import RotatedSingleStageDetector
from mmrotate.core import rbbox2result

@ROTATED_DETECTORS.register_module()
class RotatedDETR(RotatedSingleStageDetector):
    def __init__(self,
                 backbone,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(RotatedDETR, self).__init__(backbone, None, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)

    
    # over-write `forward_dummy` because:
    # the forward of bbox_head requires img_metas
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        warnings.warn('Warning! MultiheadAttention in DETR does not '
                      'support flops computation! Do not use the '
                      'results in your papers!')

        batch_size, _, height, width = img.shape
        dummy_img_metas = [
            dict(
                batch_input_shape=(height, width),
                img_shape=(height, width, 3)) for _ in range(batch_size)
        ]
        x = self.extract_feat(img)
        outs = self.bbox_head(x, dummy_img_metas)
        return outs

    # over-write `onnx_export` because:
    # (1) the forward of bbox_head requires img_metas
    # (2) the different behavior (e.g. construction of `masks`) between
    # torch and ONNX model, during the forward of bbox_head
    def onnx_export(self, img, img_metas):
        """Test function for exporting to ONNX, without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        # forward of this head requires img_metas
        outs = self.bbox_head.forward_onnx(x, img_metas)
        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape

        det_bboxes, det_labels = self.bbox_head.onnx_export(*outs, img_metas)

        return det_bboxes, det_labels

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [cx, cy, w, h, a] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(RotatedSingleStageDetector, self).forward_train(img, img_metas)
        x = self.extract_feat(img)

        cost_matrix = np.asarray(x[0].cpu().detach())
        contain_nan = (True in np.isnan(cost_matrix))
        if contain_nan:
            print('Find!!!')

            for i in range(len(img_metas)):
                print('img_metas\n', img_metas[i])
                print('The image is', img_metas[i]['filename'])
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)
        return losses

    def imshow_gpu_tensor(self, tensor):
        from PIL import Image
        from torchvision import transforms
        device = tensor[0].device
        mean = torch.tensor([123.675, 116.28, 103.53])
        std = torch.tensor([58.395, 57.12, 57.375])
        mean = mean.to(device)
        std = std.to(device)
        tensor = (tensor[0].squeeze() * std[:, None, None]) + mean[:, None, None]
        tensor = tensor[0:1]
        if len(tensor.shape) == 4:
            image = tensor.permute(0, 2, 3, 1).cpu().clone().numpy()
        else:
            image = tensor.permute(1, 2, 0).cpu().clone().numpy()
        image = image.astype(np.uint8).squeeze()
        image = transforms.ToPILImage()(image)
        image = image.resize((256, 256), Image.ANTIALIAS)
        image.show(image)

    def simple_test(self, img, img_metas, rescale=False):
        cfg = self.test_cfg

        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test_bboxes(feat, img_metas, rescale=rescale)

        bbox_results = [
            rbbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list]

        return bbox_results


