experiment = dict(
    name="HA3 Hierarchical All 9 Training 1L:1L:1L:1L:1L:1L:1K:1K:1K",
    description=" All 9 dataset classes mapped to universal classes with flat model   "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/HA3/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

# Todo: update backbone pretrained checkpoint in model file

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/hierarchical/HA3_hierarchicalv1_deeplabv3plus_r101-d8_w_ohem_w_loss_weight.py',
    '../../_base_/datasets/hiera/HA3_hierarchical_all_512_512.py',
]
ignore_index = 0

## Todo: Check samples per gpu ? keep max

data = dict(samples_per_gpu=2,
            workers_per_gpu=6)


# optimizer
optimizer = dict(
    type='SGD',
    lr=0.007,
    momentum=0.9,
    weight_decay=0.0005)

## Todo: As we removed Aux heads, Is memory sufficient for FP32 ?? No
#optimizer_config = dict()

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
    max_epochs=10)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=True,
    interval=1,
    max_keep_ckpts=15,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=1, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
