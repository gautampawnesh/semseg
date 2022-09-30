experiment = dict(
    name="FMD 7 Flat Deeplabv3+  All 9 Training os8",
    description=" All 9 dataset classes mapped to universal classes with flat deeplabv3+ model  \ "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/FMD7_2"
# random seed
seed = 1
# checkpoint file to load weights from
#load_from = "/netscratch/gautam/semseg/exp_results/FMD7/training/20220704_064718/epoch_10.pth"
#load_from = "/netscratch/gautam/semseg/exp_results/FMD7/training/20220705_062238/epoch_90.pth"
#load_from = "/netscratch/gautam/semseg/exp_results/FMD7/training/20220724_155407/epoch_10.pth"
#load_from = "/netscratch/gautam/semseg/exp_results/FMD7/training/20220725_103634/epoch_9.pth"
#load_from = "/netscratch/gautam/semseg/exp_results/FMD7/training/20220726_235911/epoch_10.pth"
load_from = "/netscratch/gautam/semseg/exp_results/FMD7/training/20220920_090341/epoch_1.pth"
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/flat/flat_train_deeplabv3plus_r101-d8.py',
    '../../_base_/datasets/flat/FMD7_all_512_512_valtrain.py',
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
    decode_head=dict(ignore_index=ignore_index, num_classes=191, loss_decode=dict(
            _delete_=True,
            type="LovaszLoss",
            reduction="none",
            loss_weight=1.0
        )),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=191, loss_decode=dict(
            _delete_=True,
            type="LovaszLoss",
            reduction="none",
            loss_weight=0.4
        )),
)

# optimizer
optimizer = dict(
    type='SGD',
    #lr=0.00001,
    #lr=0.000004,
    lr=2.652e-04,
    #lr=1.043e-05,
    #lr=7.982e-06,
    momentum=0.9,
    weight_decay=0.0005,
)
optimizer_config = dict()

# learning policy
lr_config = dict(
    policy='poly',
    #power=0.9,
    power=1.2,
    min_lr=1e-8,
    by_epoch=False)

# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=80000)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=False,
    interval=8000,
    max_keep_ckpts=15,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=8000, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
