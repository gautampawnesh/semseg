experiment = dict(
    name="Cityscapes+viper+vistas+idd+bdd10k Training",
    description="Cityscapes, vistas, idd, bdd10k and VIPER classes mapped to universal classes with flat model  ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/all_six_deeplabv3plus_67c_lovasz_loss/"
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
    decode_head=dict(ignore_index=ignore_index, num_classes=67,
                     loss_decode=dict(_delete_=True, type="LovaszLoss", reduction="none", loss_weight=1.0)),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=67,
                        loss_decode=dict(_delete_=True, type="LovaszLoss", reduction="none", loss_weight=0.4)),
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
    power=1.0,
    min_lr=0.0,
    by_epoch=False,
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
)

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=25)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=2,
    max_keep_ckpts=5,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="1_mIoU")

log_level = 'INFO'
