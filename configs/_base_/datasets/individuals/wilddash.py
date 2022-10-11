
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapAnnotations'),
    dict(type='Resize', img_scale=[
        (1920, 1208),
        (1024, 1024),
        (2400, 1510),
        (1680, 1057),
        (2400, 1510),
        (2880, 1812),
        (3360, 2114)],
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
        img_scale=(2048, 1024),
        #img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        #flip=True,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data_seed = 1


wild_data = dict(
    train=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        img_meta_data="/netscratch/gautam/semseg/raw_data_metadata/wilddash_raw_train_dataset.csv",
        is_color_to_uni_class_mapping=False,
        dataset_name="wilddash",
        num_samples=100000,
        is_universal_network=False,
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        img_meta_data="/netscratch/gautam/semseg/raw_data_metadata/wilddash_raw_val_dataset.csv",
        seg_map_suffix='.png',
        is_color_to_uni_class_mapping=False,
        test_mode=True,
        is_universal_network=False,
        num_samples=100,
        dataset_name="wilddash",
        pipeline=test_pipeline),
    test=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        img_meta_data="/netscratch/gautam/semseg/raw_data_metadata/wilddash_raw_val_dataset.csv",
        seg_map_suffix='.png',
        is_universal_network=False,
        is_color_to_uni_class_mapping=False,
        dataset_name="wilddash",
        pipeline=test_pipeline))


data = dict(
    train=wild_data["train"],
    val=wild_data["val"],
    test=wild_data["test"]
)
