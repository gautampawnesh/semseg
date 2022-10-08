experiment = dict(
    name="All 9 Training",
    description=" All 9 dataset classes mapped to universal classes with flat model 2nodes 12 gpu "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/all_nine_deeplabv3plus_191c_2/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = "/netscratch/gautam/semseg/exp_results/all_nine_deeplabv3plus_191c_2/training/20220620_230200/epoch_30.pth"
# checkpoint file to resume from
# 2 node 12gpu
# resume_from = "/netscratch/gautam/semseg/exp_results/all_nine_deeplabv3plus_191c_2/training/20220620_230200/epoch_13.pth"
resume_from = None #"/netscratch/gautam/semseg/exp_results/all_nine_deeplabv3plus_191c_2/training/20220620_230200/epoch_30.pth"

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/deeplabv3plus_r101-d8_large_features.py',
    '../../_base_/datasets/all_512_512.py',
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
    lr=5.637e-04,
    momentum=0.9,
    weight_decay=0.0004)

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
