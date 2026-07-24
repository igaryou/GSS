_base_ = [
    '../_base_/datasets/cityscapes_256x512.py',
    '../_base_/models/gss-ff_r101.py',
    '../_base_/default_runtime.py'
]

# Keep the official GSS-FF R101 model unchanged apart from the configured
# input size and the requested whole-image inference mode.
model = dict(
    decode_head=dict(img_size=(256, 512)),
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

# tools/train.py sets drop_last=True. With 2,975 images, batch size 4 and one
# rank, len(train_dataloader) is floor(2975 / 4) = 743 iterations per epoch.
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
    'gss-ff_r101_256x512_b4_800ep_cityscapes')
