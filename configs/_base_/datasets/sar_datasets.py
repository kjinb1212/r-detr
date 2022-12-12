# dataset settings
dataset_type = 'SARDatasets'
data_root = '/media/data1/Rotated_SAR/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(512, 512)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(512, 512),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # ann_file=data_root + 'trainval1024_ms/DOTA_trainval1024_ms.json',
        ann_file=data_root + 'train800/labelTxt/',
        img_prefix=data_root + 'train800/images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'val800/labelTxt/',
        img_prefix=data_root + 'val800/images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/media/data1/SSDD_DOTA_format/test/labelTxt/',
        img_prefix='/media/data1/SSDD_DOTA_format/test/images/',
        # ann_file='/media/data1/HRSID/test/labelTxt/',
        # img_prefix='/media/data1/HRSID/test/images/',
        # ann_file='/media/data1/RSDD-SAR/test/labelTxt/',
        # img_prefix='/media/data1/SAR/test/images/',
        pipeline=test_pipeline))
 