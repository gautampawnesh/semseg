experiment = dict(
    name="Hierarchical All 9 Training",
    description=" All 9 dataset classes mapped to universal classes with flat model   "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/hierarchical_deeplabv3plus_189c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/hierarchical_deeplabv3plus_r101-d8.py',
    '../_base_/datasets/all_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=4,
            workers_per_gpu=6)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=189),
    auxiliary_head=dict(ignore_index=ignore_index, num_classes=189),
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=0.01,
    momentum=0.9,
    weight_decay=0.0005)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=True)

# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=10000)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=50,
    max_keep_ckpts=5,
    create_symlink=False
)

evaluation = dict(_delete_=True, start=500, interval=500, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
