samples_per_gpu = 4

_base_ = [
    './sparse_detr_r50_10_hrsid.py'
]
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth')
    ),
    neck=dict(
        in_channels=[192, 384, 768],),
)
data = dict(
    samples_per_gpu=samples_per_gpu)

auto_scale_lr = dict(base_batch_size=4)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='MMDetWandbHook',
         init_kwargs={
             'project': 'sar_datasets', 
             'entity': 'kjinb1212', 
             'name':'hrsid_dota_pretrained',
            #  'id': 'hrsid_dota_pretrained',
             'config':{'lr':2e-4, 'batch':4},
             'resume':'allow'
        },
         log_checkpoint=True,
         log_checkpoint_metadata=True,
         num_eval_images=0)

    ])

work_dir='/media/data1/jinbeom/sar_datasets/swint10_hrsid_dota_pretrained'

load_from='/media/data1/jinbeom/sparse/sparse_swint_10_batch32_modi3_dota1_ms_1024_512/best_mAP_epoch_49.pth'