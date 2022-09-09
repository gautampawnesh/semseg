experiment = dict(
    name="FMD 8 Flat Deeplabv3+  All 9 Training os8 1L:1L:1L:1L:1L:1L:1K:1K:1K",
    description=" All 9 dataset classes mapped to universal classes with flat deeplabv3+ model  \ "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/FMD8/"
# random seed das
seed = 1
# checkpoint file to load weights from
#load_from = "/netscratch/gautam/semseg/final_weights/FMD8/training/20220801_000333/epoch_10.pth"
#load_from = "/netscratch/gautam/semseg/final_weights/FMD8/training/20220824_110833/epoch_6.pth"
load_from = None  #"/netscratch/gautam/semseg/final_weights/FMD8/training/20220829_140628/epoch_6.pth"
# checkpoint file to resume from
resume_from = "/netscratch/gautam/semseg/final_weights/FMD8/training/20220831_073718/epoch_9.pth"

_base_ = [
    '../../_base_/default_runtime.py',
    '../../_base_/models/flat/FMD8_flat_train_deeplabv3plus_r101-d8.py',
    '../../_base_/datasets/flat/FMD8_all_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=8,   ### TODO: Should be similar to HA
            workers_per_gpu=6)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=191)
)

# optimizer
optimizer = dict(
    type='SGD',
    #lr=0.007,
    #lr=0.0007,
    #lr=0.0004,
    lr=1.810e-04,
    momentum=0.9,
    weight_decay=0.0005
)
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