experiment = dict(
    name="Cityscapes Training",
    description="Sample  Iter based training: Cityscapes classes mapped to universal classes with flat model /",
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/exp_results/h_deeplab_city_80K_191c_fp16_v1/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/hierarchical_deeplabv3plus_r101-d8_w_ohem.py',
    '../_base_/datasets/cityscapes_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=2,
            workers_per_gpu=6,
            test=dict(ignore_index=ignore_index),
            train=dict(ignore_index=ignore_index),
            val=dict(ignore_index=ignore_index))


# optimizer
optimizer = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9,
    weight_decay=0.0004)
optimizer_config = dict(type="Fp16OptimizerHook", loss_scale=512.0)
# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-4,
    by_epoch=False)
# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=80000)   # 160000
# checkpoints settings
checkpoint_config = dict(
    by_epoch=False,
    interval=4000,
    max_keep_ckpts=5,
    create_symlink=False
)
evaluation = dict(_delete_=True, interval=4000, metric="mIoU", pre_eval=True, save_best="mIoU")
log_level = 'INFO'
