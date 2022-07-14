
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
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=True,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

extra_class_mapping = {0: [0, 184], 1: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 181, 182, 186],
                       2: [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 44, 47, 53, 97, 99, 128, 180, 190],
                       3: [27, 28, 29, 30, 31, 32, 33, 74, 89, 93, 96, 98, 104, 121, 126, 130, 135, 136, 138, 152, 167],
                       4: [38, 39, 124, 40, 43, 45, 55, 185, 154, 71, 134, 42, 41, 123, 46, 54, 49, 48, 50, 52, 61],
                       5: [183, 115, 36, 177, 35, 37, 187, 188, 112, 78, 82, 25, 24, 26, 105, 157, 75, 145, 141, 34],
                       6: [189, 60, 57, 58, 59, 56],
                       7: [64, 66, 67, 68, 70, 72, 73, 79, 81, 90, 91, 83, 108, 131, 142, 179, 106, 88, 118, 148, 100, 80, 51, 65,
                           86, 102, 84, 77, 76, 69, 110, 107, 160, 129, 94, 63, 62, 163,
                           85, 109, 122, 171, 178,
                           176, 175, 174, 172, 173, 169, 166, 165, 164, 156, 147, 146, 144, 143, 140, 137, 125, 111, 101, 150, 153, 161, 87,
                           170, 168, 159, 127, 120, 117, 103,
                           162, 158, 155, 151, 149, 139, 133, 132, 119, 116, 114, 113, 95, 92]}

vistas_data = dict(
    train=dict(
        type='VistasDataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='training/images',
        ann_dir='training/v1.2/labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        num_samples=10000,
        is_extra_class_mapping=True,
        extra_class_map=extra_class_mapping,
        pipeline=train_pipeline),
    val=dict(
        type='VistasDataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='validation/images',
        ann_dir='validation/v1.2/labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        is_extra_class_mapping=True,
        extra_class_map=extra_class_mapping,
        num_samples=500,
        test_mode=True,
        pipeline=val_pipeline),
    test=dict(
        type='VistasDataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='validation/images',
        ann_dir='validation/v1.2/labels',
        seg_map_suffix='.png',
        class_color_mode="RGB",
        is_extra_class_mapping=True,
        extra_class_map=extra_class_mapping,
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        pipeline=test_pipeline))

ade_data = dict(
    train=dict(
        type='UniversalAdeDataset',
        data_root='/ds-av/public_datasets/ade_20k/raw',
        img_dir='images/training',
        ann_dir='annotations/training',
        seg_map_suffix='.png',
        is_extra_class_mapping=True,
        extra_class_map=extra_class_mapping,
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/ade20k_class_mapping.csv",
        is_color_to_uni_class_mapping=False,
        dataset_name="ade",
        num_samples=10000,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalAdeDataset',
        data_root='/ds-av/public_datasets/ade_20k/raw',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/ade20k_class_mapping.csv",
        seg_map_suffix='.png',
        is_extra_class_mapping=True,
        extra_class_map=extra_class_mapping,
        is_color_to_uni_class_mapping=False,
        dataset_name="ade",
        num_samples=500,
        test_mode=True,
        pipeline=val_pipeline),
    test=dict(
        type='UniversalAdeDataset',
        data_root='/ds-av/public_datasets/ade_20k/raw',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/ade20k_class_mapping.csv",
        seg_map_suffix='.png',
        is_extra_class_mapping=True,
        extra_class_map=extra_class_mapping,
        is_color_to_uni_class_mapping=False,
        dataset_name="ade",
        pipeline=test_pipeline))


data = dict(
    train=dict(
        type="CustomConcatDataset",
        datasets=[vistas_data["train"], ade_data["train"]],
    ),
    val=dict(
        type="CustomConcatDataset",
        datasets=[vistas_data["val"], ade_data["val"]]
    ),
    test=vistas_data["test"]
)
