experiment = dict(
    name="Cityscapes Training",
    description="Baseline 0: Cityscapes classes mapped to universal classes with flat model ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/baseline_flat/cityscapes_deeplabv3plus_189c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv3plus_r101-d8.py',
    '../_base_/datasets/cityscapes_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=4,
            workers_per_gpu=6,
            test=dict(ignore_index=ignore_index),
            train=dict(ignore_index=ignore_index),
            val=dict(ignore_index=ignore_index))

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=189),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=189),
)

# optimizer
optimizer = dict(
    type='SGD', #   lr=0.01, # 1- 100 epochs lr=1.585e-04,
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
    max_epochs=100)
# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)
evaluation = dict(_delete_=True, start=1, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")
log_level = 'INFO'
