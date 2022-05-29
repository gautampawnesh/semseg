experiment = dict(
    name="Cityscapes+viper+vistas+idd+bdd10k+gta5 Training",
    description="Cityscapes, vistas, idd, bdd10k, gta5 and VIPER classes mapped to universal classes with flat model  ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/cityscapes_viper_idd_bdd_gta5_vistas_deeplabv3plus_66c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_viper_vistas_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=4,
            workers_per_gpu=8)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=67),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=67),
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
    by_epoch=False)

# runtime settings
runner = dict(type='IterBasedRunner', max_iters=25)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=False,
    interval=5,
    max_keep_ckpts=5,
    create_symlink=False
)

# evaluation = dict(_delete_=True, interval=25, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="0_mIoU")
evaluation = dict(_delete_=True, interval=25, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="0_mIoU")

log_level = 'INFO'
