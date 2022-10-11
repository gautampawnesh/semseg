# dataset settings
# gt has label ids : https://github.com/srrichter/viper/blob/master/classes.csv

dataset_type = 'UniversalViperDataset'
data_root = '/ds-av/public_datasets/viper/raw'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapAnnotations'),
    dict(type='Resize', img_scale=[(1920, 1208), (768, 768), (1024, 1024), (512, 1024), (2400, 1510),
                                   (960, 604), (2880, 1812), (3360, 2114)],
         ratio_range=None, multiscale_mode="value"),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1920, 1208),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='train/img',
        ann_dir='train/cls',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
        dataset_name="viper",
        ignore_index=0,  # gt are Label ids
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/img',
        ann_dir='val/cls',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
        dataset_name="viper",
        ignore_index=0,  # gt are Label ids
        num_val_samples=1000,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='val/img',
        ann_dir='val/cls',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
        dataset_name="viper",
        ignore_index=0,  # gt are Label ids
        num_val_samples=1000,
        pipeline=test_pipeline))
