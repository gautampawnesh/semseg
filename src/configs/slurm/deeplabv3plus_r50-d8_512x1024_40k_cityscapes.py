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
work_dir = "/netscratch/gautam/semseg/results"
dist_params = dict(backend="nccl")
