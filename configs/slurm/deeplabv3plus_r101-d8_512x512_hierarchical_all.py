experiment = dict(
    name="Hierarchical All 9 Training",
    description=" All 9 dataset classes mapped to universal classes with flat model   "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/hierarchical_deeplabv3plus_191c/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = "/netscratch/gautam/semseg/exp_results/hierarchical_deeplabv3plus_191c/training/20220623_112916/epoch_6.pth"
# checkpoint file to resume from
resume_from = None #"/netscratch/gautam/semseg/exp_results/hierarchical_deeplabv3plus_191c/training/20220622_100800/epoch_2.pth"

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/hierarchical_deeplabv3plus_r101-d8.py',
    '../_base_/datasets/hierarchical_all_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=1,
            workers_per_gpu=6)


# optimizer
optimizer = dict(
    type='SGD',
    lr=4.440e-03,
    momentum=0.9,
    weight_decay=0.0005)
#optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
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
