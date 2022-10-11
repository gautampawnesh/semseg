experiment = dict(
    name="ADE Individuals ",
    description=" Deeplabv3+  with 1L samples "
)
# checkpoint_file="/netscratch/gautam/semseg/final_weights/ade/training/20221007_185032/epoch_9.pth"
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/ade"
# random seed das
seed = 1
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/individuals/deeplabv3plus_r101-d8.py',
    '../_base_/datasets/individuals/ade.py',
]
ignore_index = 0

data = dict(samples_per_gpu=8,
            workers_per_gpu=6)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=151)
)
log_level = 'INFO'
