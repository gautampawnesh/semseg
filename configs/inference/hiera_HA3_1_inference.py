experiment = dict(
    name="HA3 Inference",
    description=" All 9 dataset classes mapped to universal classes with flat model   "
)
#checkpoint= "/netscratch/gautam/semseg/final_weights/HA3_1/training/20220820_082605/epoch_10.pth"

# directory to save logs and models
work_dir = "/netscratch/gautam/semseg/final_weights/HA3_1/"
# random seed
seed = 1
# checkpoint file to load weights from
load_from = None
# checkpoint file to resume from
resume_from = None

# Todo: update backbone pretrained checkpoint in model file

_base_ = [
    '../_base_/default_runtime.py',
    '../_base_/models/hierarchical/HA3_1_hierarchicalv1_deeplabv3plus_r101-d8_w_ohem_w_loss_weight.py',
    '../_base_/datasets/hiera/HA3_hierarchical_all_512_512.py',
]
ignore_index = 0

## Todo: Check samples per gpu ? keep max

data = dict(samples_per_gpu=2,
            workers_per_gpu=6)

log_level = 'INFO'