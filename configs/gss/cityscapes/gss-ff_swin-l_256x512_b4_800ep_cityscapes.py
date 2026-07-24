_base_ = [
    '../_base_/datasets/cityscapes_256x512.py',
    '../_base_/models/gss-ff_swin-l.py',
    '../_base_/default_runtime.py'
]

# Keep the official Cityscapes colors, including the final ignore color.
cityscapes_palette = [
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
    [0, 80, 100], [0, 0, 230], [119, 11, 32], [0, 0, 0]
]

model = dict(
    decode_head=dict(
        img_size=(256, 512),
        num_classes=19,
        palette=cityscapes_palette),
    test_cfg=dict(_delete_=True, mode='whole'))

optimizer = dict(
    type='AdamW',
    lr=0.0015,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict()

# tools/train.py uses drop_last=True:
# floor(2,975 Cityscapes train images / batch 4) = 743 iterations per epoch.
iterations_per_epoch = 743
max_epochs = 800
runner = dict(
    type='IterBasedRunner',
    max_iters=iterations_per_epoch * max_epochs)

interval_50_epochs = iterations_per_epoch * 50
checkpoint_config = dict(
    by_epoch=False,
    interval=interval_50_epochs,
    max_keep_ckpts=5)
evaluation = dict(
    interval=interval_50_epochs,
    metric='mIoU',
    pre_eval=True)
log_config = dict(interval=50)

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)

work_dir = (
    '/home/igarashi_25/GSS/work_dirs/'
    'gss-ff_swin-l_256x512_b4_800ep_cityscapes')
