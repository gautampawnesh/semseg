experiment = dict(
    name="ade Training",
    description="Baseline 0: ade classes mapped to universal classes with flat model  ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/baseline_flat/ade_deeplabv3plus_187c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = "/netscratch/gautam/semseg/baseline_flat/ade_deeplabv3plus_187c/training/20220605_164246/best_mIoU_epoch_200.pth"
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv3plus_r101-d8.py',
    '../../_base_/datasets/ade_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=4,
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
    lr=1.841e-04,
    momentum=0.9,
    weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=0.0,
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
evaluation = dict(_delete_=True, start=50, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")
log_level = 'INFO'
