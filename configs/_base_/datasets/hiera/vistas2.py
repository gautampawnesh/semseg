
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)

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


vistas_data = dict(
    train=dict(
        type='Vistas2Dataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='training/images',
        ann_dir='training/v2.0/labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas2.csv",
        img_meta_data="/netscratch/gautam/semseg/raw_data_metadata/vistas_raw_train_dataset.csv",
        dataset_name="vistas2",
        ignore_index=123,
        is_universal_network=False,
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='Vistas2Dataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='validation/images',
        ann_dir='validation/v2.0/labels',
        seg_map_suffix='.png',
        ignore_index=123,
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas2.csv",
        img_meta_data="/netscratch/gautam/semseg/raw_data_metadata/vistas_raw_val_dataset.csv",
        dataset_name="vistas2",
        num_samples=500,
        is_universal_network=False,
        test_mode=True,
        pipeline=test_pipeline),
    test=dict(
        type='Vistas2Dataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='validation/images',
        ann_dir='validation/v1.2/labels',
        seg_map_suffix='.png',
        class_color_mode="RGB",
        is_universal_network=False,
        ignore_index=123,
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas2.csv",
        img_meta_data="/netscratch/gautam/semseg/raw_data_metadata/vistas_raw_val_dataset.csv",
        dataset_name="vistas2",
        pipeline=test_pipeline))


data = dict(
    train=vistas_data["train"],
    val=vistas_data["val"],
    test=vistas_data["test"]
)
