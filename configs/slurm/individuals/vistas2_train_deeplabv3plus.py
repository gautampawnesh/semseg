experiment = dict(
    name="Vistas2.0 Individuals ",
    description=" Deeplabv3+  "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/vistas2"
# random seed das
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/individuals/mvp2_deeplabv3plus_r101.py',
    '../../_base_/datasets/individuals/vistas2.py',
]
ignore_index = 123

data = dict(samples_per_gpu=2,
            workers_per_gpu=6)

model = dict(pretrained="https://assets-1257038460.cos.ap-beijing.myqcloud.com/resnet101_v1d.pth",
             decode_head=dict(ignore_index=ignore_index, num_classes=124)
)

# optimizer
optimizer = dict(
    type='SGD',
    lr=1e-2,
    momentum=0.9,
    weight_decay=1e-4
)
optimizer_config = dict() #dict(type="Fp16OptimizerHook", loss_scale=512.0)

# learning policy
lr_config = dict(
    policy='poly',
    power=0.9,
    min_lr=1e-5,
    by_epoch=False)

# runtime settings
runner = dict(
    type='IterBasedRunner',
    max_iters=240000)

# checkpoints settings
checkpoint_config = dict(
    by_epoch=False,
    interval=16000,
    max_keep_ckpts=15,
    create_symlink=False
)

evaluation = dict(_delete_=True, interval=16000, metric="mIoU", gpu_collect=True, pre_eval=True, save_best="mIoU")

log_level = 'INFO'
