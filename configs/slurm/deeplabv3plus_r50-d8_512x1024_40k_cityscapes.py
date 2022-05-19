experiment = dict(
    name="Cityscape Training",
    description="Example ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/results/cityscape_train_deeplabv3plus_19c/training"
# random seed
seed = 1
# launcher
launcher = "slurm"
dist_params = dict(backend="nccl")
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None
cudnn_benchmark = True

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_512_1024.py',
]
ignore_index = 255
data = dict(samples_per_gpu=8,
            workers_per_gpu=2,
            test=dict(ignore_index=ignore_index),
            train=dict(ignore_index=ignore_index),
            val=dict(ignore_index=ignore_index))

model = dict(
    decode_head=dict(ignore_index=ignore_index),
    auxiliary_head=dict(ignore_index=ignore_index),
)
workflow = [("train", 1), ("val", 1)]

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
    max_epochs=100)
# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=5,
    max_keep_ckpts=5,
    create_symlink=False
)
evaluation = dict(
    interval=1000,
    metric='mIoU',
    gpu_collect=True,
    pre_eval=True,
    save_best="mIoU")


log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

log_level = 'INFO'
