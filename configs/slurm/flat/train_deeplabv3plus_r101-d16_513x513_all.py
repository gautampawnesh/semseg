experiment = dict(
    name="Flat Deeplabv3+  All 9 Training",
    description=" All 9 dataset classes mapped to universal classes with flat deeplabv3+ model  \ "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/flat_all_nine_deeplabv3plus_191c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/flat/flat_train_deeplabv3plus_r101-d16.py',
    '../../_base_/datasets/flat/all_513_513.py',
]
ignore_index = 0

data = dict(samples_per_gpu=4,
            workers_per_gpu=6)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=191),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=191),
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005,
    paramwise_cfg=dict(
        custom_keys={
            ".backbone": dict(lr_mult=0.1)
        })
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
    max_epochs=60)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
