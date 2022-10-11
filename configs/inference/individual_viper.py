experiment = dict(
    name="Viper Individuals ",
    description=" Deeplabv3+  with 1L samples "
)
#checkpoint="/netscratch/gautam/semseg/final_weights/viper/training/20221007_114704/epoch_6.pth"
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/viper"
# random seed das
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/individuals/deeplabv3plus_r101-d8.py',
    '../_base_/datasets/individuals/viper.py',
]
ignore_index = 0

data = dict(samples_per_gpu=8,   ### TODO: Should be similar to HA
            workers_per_gpu=6)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=23)
)
log_level = 'INFO'