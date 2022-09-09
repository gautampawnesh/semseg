experiment = dict(
    name="Inference FMD 7 Flat Deeplabv3+  All 9 Training os8",
    description=" All 9 dataset classes mapped to universal classes with flat deeplabv3+ model  \ "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/benchmark/FMD7_test"
# random seed
seed = 1
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/flat/flat_train_deeplabv3plus_r101-d8.py',
    '../_base_/datasets/flat/FMD7_all_1024_1024.py',
]
ignore_index = 0

data = dict(samples_per_gpu=2,
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
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=1.0,
            avg_non_ignore=True
        )),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=191, loss_decode=dict(
            _delete_=True,
            type="CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=0.4,
            avg_non_ignore=True
        )),
)

# optimizer
evaluation = dict(_delete_=True)

log_level = 'INFO'
