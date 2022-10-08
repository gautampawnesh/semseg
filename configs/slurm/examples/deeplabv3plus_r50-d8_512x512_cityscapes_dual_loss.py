experiment = dict(
    name="Cityscapes Training with dual loss",
    description="Cityscapes classes mapped to universal classes with flat model  ",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/cityscapes_deeplabv3plus_67c_dual_loss/"
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
    decode_head=dict(ignore_index=ignore_index, num_classes=67,
                     loss_decode=[
                        dict(type="LovaszLoss", reduction="none", loss_weight=1.0),
                        dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                     ]),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=67,
                        loss_decode=[
                            dict(type="LovaszLoss", reduction="none", loss_weight=0.4),
                            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)
                        ]),
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
    max_epochs=100)
# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)

log_level = 'INFO'
evaluation = dict(interval=20, metric="mIoU", gpu_collect=True, pre_eval=True, save_best='mIoU')
