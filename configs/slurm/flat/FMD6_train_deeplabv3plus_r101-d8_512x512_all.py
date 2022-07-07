experiment = dict(
    name="FMD6 Flat Deeplabv3+  All 9 Training os8",
    description=" All 9 dataset classes mapped to universal classes with flat deeplabv3+ model  \ "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/FMD6"
# random seed
seed = 1
# checkpoint file to load weights from
# FMD1 pretrained
#load_from = "/netscratch/gautam/semseg/exp_results/all_nine_deeplabv3plus_189c/training/20220612_074736/best_mIoU_epoch_50.pth"
load_from = "/netscratch/gautam/semseg/exp_results/FMD5/training/20220630_124541/epoch_35.pth"
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/flat/flat_train_deeplabv3plus_r101-d8.py',
    '../../_base_/datasets/flat/all_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=4,
            workers_per_gpu=4)

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
    decode_head=dict(ignore_index=ignore_index, num_classes=191, loss_decode=dict(_delete_=True,
            type="MultiClassFocalLoss",
            use_sigmoid=False,
            gamma=2.0,
            alpha=1.0,
            loss_weight=1.0,
        )),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=191, loss_decode=dict(_delete_=True,
            type="MultiClassFocalLoss",
            use_sigmoid=False,
            gamma=2.0,
            alpha=1.0,
            loss_weight=0.4,
        ))

)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0005,
)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-5,
    by_epoch=True)

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=40)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
