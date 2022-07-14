experiment = dict(
    name="HIMini-5 Flat Deeplabv3+  on Mini (vistas+ade)",
    description=" universal level 1 classes with flat deeplabv3+ model  \ "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/HIMini5"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/hierarchical_levels/HIMini1_train_deeplabv3plus_r101-d8.py',
    '../../_base_/datasets/hierarchical_levels/vistas_ade_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=4,
            workers_per_gpu=6)

model = dict(
    pretrained=None,
    backbone=dict(
        _delete_=True,
        type='ResNetV1c',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(ignore_index=ignore_index, num_classes=8, loss_decode=dict(
            _delete_=True,
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True,
            class_weight=[0.0, 1., 1., 1., 1.15, 1.0, 1.25, 1.]
        )),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=8, loss_decode=dict(
            _delete_=True,
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
            avg_non_ignore=True
        )),
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.007,
    momentum=0.9,
    weight_decay=0.0005,
)
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
    max_epochs=20)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
