experiment = dict(
    name="Cityscape Training",
    description="Example ",
)

seed = 1

_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/cityscapes_512_1024.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40e.py'
]
ignore_index=255
data = dict(test=dict(ignore_index=ignore_index), train=dict(ignore_index=ignore_index), val=dict(ignore_index=ignore_index))

model = dict(
    decode_head=dict(ignore_index=ignore_index),
    auxiliary_head=dict(ignore_index=ignore_index),
)

work_dir = "/netscratch/gautam/semseg/results"
dist_params = dict(backend="nccl")
