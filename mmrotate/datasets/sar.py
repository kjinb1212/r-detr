# Copyright (c) OpenMMLab. All rights reserved.
from .builder import ROTATED_DATASETS
from .dota import DOTADataset
import glob
import os
import os.path as osp
import re
import tempfile
import time
import zipfile
from collections import defaultdict
from functools import partial
from multiprocessing import get_context

import mmcv
import numpy as np
import torch
from mmcv.ops import nms_rotated
from mmdet.datasets.custom import CustomDataset
from mmrotate.core import eval_rbbox_map, obb2poly_np, poly2obb_np


@ROTATED_DATASETS.register_module()
class SARDataset(DOTADataset):
    """SAR ship dataset for detection (Support RSSDD and HRSID)."""
    CLASSES = ('ship', )
    PALETTE = [
        (0, 255, 0),
    ]

    def load_annotations(self, ann_folder, ext='.jpg'):
        """
            Args:
                ann_folder: folder that contains xywha format txt files
        """
        cls_map = {c: i
                   for i, c in enumerate(self.CLASSES)
                   }  # in mmdet v2.0 label is 0-based
        ann_files = glob.glob(ann_folder + '/*.txt')
        
        data_infos = []
        if not ann_files:  # test phase
            ann_files = glob.glob(ann_folder + '/*'+ ext)
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + ext
                data_info['filename'] = img_name
                data_info['ann'] = {}
                data_info['ann']['bboxes'] = []
                data_info['ann']['labels'] = []
                data_infos.append(data_info)
        else:
            for ann_file in ann_files:
                data_info = {}
                img_id = osp.split(ann_file)[1][:-4]
                img_name = img_id + ext
                data_info['filename'] = img_name
                data_info['ann'] = {}
                gt_bboxes = []
                gt_labels = []
                gt_bboxes_ignore = []
                gt_labels_ignore = []

                if os.path.getsize(ann_file) == 0:
                    continue

                with open(ann_file) as f:
                    s = f.readlines()
                    for si in s:
                        bbox_info = si.split()
                        x, y, w, h, theta = np.array(bbox_info[:5], dtype=np.float32)
                        if theta == 0.0:
                            tmp = w
                            w = h
                            h = tmp
                            theta = 90.0
                        a = theta / 180 * np.pi
                        assert 0 < a <= np.pi / 2, 'ann_file:{}, theta:{}, a:{}, np.pi/2={}'.format(ann_file, theta, a, np.pi/2)

                        cls_name = bbox_info[5]
                        difficulty = int(bbox_info[6])
                        label = cls_map[cls_name]
                        if difficulty > self.difficulty:
                            pass
                        else:
                            gt_bboxes.append([x, y, w, h, a])
                            gt_labels.append(label)

                if gt_bboxes:
                    data_info['ann']['bboxes'] = np.array(
                        gt_bboxes, dtype=np.float32)
                    data_info['ann']['labels'] = np.array(
                        gt_labels, dtype=np.int64)

                else:
                    data_info['ann']['bboxes'] = np.zeros((0, 5),
                                                          dtype=np.float32)
                    data_info['ann']['labels'] = np.array([], dtype=np.int64)


                if gt_bboxes_ignore:
                    data_info['ann']['bboxes_ignore'] = np.array(
                        gt_bboxes_ignore, dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        gt_labels_ignore, dtype=np.int64)
                else:
                    data_info['ann']['bboxes_ignore'] = np.zeros(
                        (0, 5), dtype=np.float32)
                    data_info['ann']['labels_ignore'] = np.array(
                        [], dtype=np.int64)

                data_infos.append(data_info)

        self.img_ids = [*map(lambda x: x['filename'][:-4], data_infos)]
        return data_infos