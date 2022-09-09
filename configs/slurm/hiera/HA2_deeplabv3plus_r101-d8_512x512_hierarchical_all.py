experiment = dict(
    name="HA2 Hierarchical All 9 Training",
    description=" All 9 dataset classes mapped to universal classes with flat model   "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/HA2/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
#resume_from = "/netscratch/gautam/semseg/exp_results/HA2/training/20220709_171511/epoch_12.pth"
resume_from = "/netscratch/gautam/semseg/exp_results/HA2/training/20220709_171511/epoch_24.pth"

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/hierarchical/HA2_hierarchicalv1_deeplabv3plus_r101-d8_w_ohem_w_loss_weight.py',
    '../../_base_/datasets/hiera/HA2_hierarchical_all_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=1,
            workers_per_gpu=6)


# optimizer
optimizer = dict(
    type='SGD',
    lr=0.007,
    momentum=0.9,
    weight_decay=0.0005)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)


# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-5,
    by_epoch=True)

# runtime settings
runner = dict(
    type='EpochBasedRunner',
    max_epochs=50)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=5,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
