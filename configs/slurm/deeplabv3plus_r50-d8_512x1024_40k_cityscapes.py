experiment = dict(
    name="Cityscape Training",
    description="Example ",
)

seed = 1

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_512_1024.py',
]
ignore_index=255
data = dict(samples_per_gpu=8,
            workers_per_gpu=2,
            test=dict(ignore_index=ignore_index), train=dict(ignore_index=ignore_index), val=dict(ignore_index=ignore_index))

model = dict(
    decode_head=dict(ignore_index=ignore_index),
    auxiliary_head=dict(ignore_index=ignore_index),
)
workflow = [("train", 1), ("val", 1)]
work_dir = "/netscratch/gautam/semseg/results/cityscape_train_deeplabv3plus_19c"
dist_params = dict(backend="nccl")

# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()
# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)
# runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(by_epoch=True, interval=5)
evaluation = dict(interval=1000, metric='mIoU', save_best="mIoU")



log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=True),
        dict(type='TensorboardLoggerHook')
    ])

log_level = 'INFO'
load_from = None
resume_from = None
cudnn_benchmark = True