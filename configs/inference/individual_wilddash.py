experiment = dict(
    name="Wilddash Individuals ",
    description=" Deeplabv3+  with 1L samples "
)
#checkpoint="/netscratch/gautam/semseg/final_weights/wilddash/training/20221007_193755/epoch_10.pth"
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/wilddash"
# random seed das
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/individuals/deeplabv3plus_r101-d8.py',
    '../_base_/datasets/individuals/wilddash.py',
]
ignore_index = 0

data = dict(samples_per_gpu=8,
            workers_per_gpu=6)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=26)
)
log_level = 'INFO'