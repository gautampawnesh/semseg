experiment = dict(
    name="Cityscapes Training",
    description="Sample Iter based training: Cityscapes classes mapped to universal classes with flat model /",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/cityscapes_iter_deeplabv3plus_189c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = "/netscratch/gautam/semseg/exp_results/cityscapes_iter_deeplabv3plus_189c/training/20220614_103215/best_mIoU_iter_80000.pth"
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
    type='SGD',
    lr=1.004e-04,
    momentum=0.9,
    weight_decay=0.0004)
optimizer_config = dict()
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=False)
# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=80000)   # 160000
# checkpoints settings
checkpoint_config = dict(
    by_epoch=False,
    interval=8000,
    max_keep_ckpts=5,
    create_symlink=False
)
evaluation = dict(_delete_=True, interval=8000, metric="mIoU", pre_eval=True, save_best="mIoU")
log_level = 'INFO'
