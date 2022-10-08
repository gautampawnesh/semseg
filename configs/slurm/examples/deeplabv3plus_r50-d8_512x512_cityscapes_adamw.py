experiment = dict(
    name="Cityscapes Training",
    description="Cityscapes classes mapped to universal classes with flat model  ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/cityscapes_deeplabv3plus_adamw_67c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv3plus_r50-d8.py',
    '../../_base_/datasets/cityscapes_512_512.py',
]
ignore_index = 255

data = dict(samples_per_gpu=4,
            workers_per_gpu=8,
            test=dict(ignore_index=ignore_index),
            train=dict(ignore_index=ignore_index),
            val=dict(ignore_index=ignore_index))

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=67),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=67),
)


# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optimizer = dict(
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)
optimizer_config = dict()

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=100)
# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)

log_level = 'INFO'
