experiment = dict(
    name="Scannet Training",
    description="Scannet classes mapped to universal classes with flat model  ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/baseline1/scannet_deeplabv3plus_187c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv3plus_r101-d8.py',
    '../_base_/datasets/scannet_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=8,
            workers_per_gpu=8,
            test=dict(ignore_index=ignore_index),
            train=dict(ignore_index=ignore_index),
            val=dict(ignore_index=ignore_index))

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=187),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=187),
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=True)
# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=200)
# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)

log_level = 'INFO'
