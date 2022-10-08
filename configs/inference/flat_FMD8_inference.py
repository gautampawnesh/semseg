experiment = dict(
    name="FMD 8 Flat Deeplabv3+  Inference",
    description=" All 9 dataset classes mapped to universal classes with flat deeplabv3+ model  \ "
)
# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/FMD8/"
# random seed das
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/flat/FMD8_flat_train_deeplabv3plus_r101-d8.py',
    '../_base_/datasets/flat/FMD8_all_512_512.py',
]
ignore_index = 0

data = dict(samples_per_gpu=8,   ### TODO: Should be similar to HA
            workers_per_gpu=6)

model = dict(
    decode_head=dict(ignore_index=ignore_index, num_classes=191)
)

log_level = 'INFO'