
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 1024)
data_seed = 1

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
city_data = dict(
    train=dict(
        type="UniversalCityscapesDataset",
        data_root='/ds-av/public_datasets/cityscapes/raw',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        seg_map_suffix='_gtFine_labelIds.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
        ignore_index=0,  # gt has Labelids
        num_samples=100000,
        data_seed=data_seed,
        dataset_name="cityscapes",
        pipeline=train_pipeline),
    val=dict(
        type="UniversalCityscapesDataset",
        data_root='/ds-av/public_datasets/cityscapes/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
        seg_map_suffix='_gtFine_labelIds.png',
        ignore_index=0,  # gt has Labelids
        dataset_name="cityscapes",
        test_mode=True,
        pipeline=val_pipeline),
    test=dict(
        type="UniversalCityscapesDataset",
        data_root='/ds-av/public_datasets/cityscapes/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
        seg_map_suffix='_gtFine_labelIds.png',
        ignore_index=0,  # gt has Labelids
        dataset_name="cityscapes",
        pipeline=test_pipeline))

viper_data = dict(
    train=dict(
        type="UniversalViperDataset",
        data_root='/ds-av/public_datasets/viper/raw',
        img_dir='train/img',
        ann_dir='train/cls',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
        dataset_name="viper",
        ignore_index=0,  # gt are Label ids
        num_samples=100000,
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type="UniversalViperDataset",
        data_root='/ds-av/public_datasets/viper/raw',
        img_dir='val/img',
        ann_dir='val/cls',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
        dataset_name="viper",
        ignore_index=0,  # gt are Label ids
        num_samples=500,
        test_mode=True,
        pipeline=val_pipeline),
    test=dict(
        type="UniversalViperDataset",
        data_root='/ds-av/public_datasets/viper/raw',
        img_dir='val/img',
        ann_dir='val/cls',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
        dataset_name="viper",
        ignore_index=0,  # gt are Label ids
        pipeline=test_pipeline))

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
        num_samples=100000,
        data_seed=data_seed,
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
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/ade20k_class_mapping.csv",
        is_color_to_uni_class_mapping=False,
        dataset_name="ade",
        num_samples=100000,
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalAdeDataset',
        data_root='/ds-av/public_datasets/ade_20k/raw',
        img_dir='images/validation',
        ann_dir='annotations/validation',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/ade20k_class_mapping.csv",
        seg_map_suffix='.png',
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
        is_color_to_uni_class_mapping=False,
        dataset_name="ade",
        pipeline=test_pipeline))

wild_data = dict(
    train=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        is_color_to_uni_class_mapping=False,
        dataset_name="wilddash",
        num_samples=100000,
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        seg_map_suffix='.png',
        is_color_to_uni_class_mapping=False,
        test_mode=True,
        dataset_name="wilddash",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        seg_map_suffix='.png',
        is_color_to_uni_class_mapping=False,
        dataset_name="wilddash",
        pipeline=test_pipeline))

scannet_data = dict(
    train=dict(
        type='UniversalScannetDataset',
        data_root='/netscratch/gautam/scannet/',
        img_dir='scans',
        ann_dir='scans',
        seg_map_suffix='_labelId.png',
        split="/netscratch/gautam/scannet/scannet_train.json",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/scannet_class_mapping.csv",
        is_color_to_uni_class_mapping=False,
        dataset_name="scannet",
        num_samples=100000,
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalScannetDataset',
        data_root='/netscratch/gautam/scannet/',
        img_dir='scans',
        ann_dir='scans',
        split="/netscratch/gautam/scannet/scannet_val.json",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/scannet_class_mapping.csv",
        seg_map_suffix='_labelId.png',
        num_samples=500,
        test_mode=True,
        is_color_to_uni_class_mapping=False,
        dataset_name="scannet",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalScannetDataset',
        data_root='/netscratch/gautam/scannet/',
        img_dir='scans',
        ann_dir='scans',
        split="/netscratch/gautam/scannet/scannet_val.json",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/scannet_class_mapping.csv",
        seg_map_suffix='_labelId.png',
        is_color_to_uni_class_mapping=False,
        dataset_name="scannet",
        pipeline=test_pipeline))

idd_data = dict(
    train=dict(
        type='UniversalIddDataset',
        data_root='/netscratch/gautam/idd_segmentation/raw',
        img_dir='leftImg8bit/train',
        ann_dir='gtFine/train',
        dataset_name="idd",
        num_samples=100,
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
        pipeline=train_pipeline),
    val=dict(
        type='UniversalIddDataset',
        data_root='/netscratch/gautam/idd_segmentation/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        dataset_name="idd",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalIddDataset',
        data_root='/netscratch/gautam/idd_segmentation/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        dataset_name="idd",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
        pipeline=test_pipeline)
)

gta_data = dict(
    train=dict(
        type='UniversalPlayingForDataDataset',
        data_root='/ds-av/public_datasets/playing_for_data/raw',
        img_dir='images',
        ann_dir='labels',
        dataset_name="playing_for_data",
        num_samples=100,
        split="/netscratch/gautam/playing_for_data/train.txt",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/playing_for_data_class_mapping.csv",
        pipeline=train_pipeline),
    val=dict(
        type='UniversalPlayingForDataDataset',
        data_root='/ds-av/public_datasets/playing_for_data/raw',
        img_dir='images',
        ann_dir='labels',
        dataset_name="playing_for_data",
        split="/netscratch/gautam/playing_for_data/val.txt",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/playing_for_data_class_mapping.csv",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalPlayingForDataDataset',
        data_root='/ds-av/public_datasets/playing_for_data/raw',
        img_dir='images',
        ann_dir='labels',
        dataset_name="playing_for_data",
        split="/netscratch/gautam/playing_for_data/val.txt",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/playing_for_data_class_mapping.csv",
        pipeline=train_pipeline),
)

bdd_data = dict(
    train=dict(
        type='UniversalBdd10kDataset',
        data_root='/ds-av/public_datasets/bdd100k/raw',
        img_dir='images/10k/train',
        ann_dir='labels/sem_seg/masks/train',
        dataset_name="bdd",
        num_samples=100,
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
        pipeline=train_pipeline),
    val=dict(
        type='UniversalBdd10kDataset',
        data_root='/ds-av/public_datasets/bdd100k/raw',
        img_dir='images/10k/val',
        ann_dir='labels/sem_seg/masks/val',
        dataset_name="bdd",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalBdd10kDataset',
        data_root='/ds-av/public_datasets/bdd100k/raw',
        img_dir='images/10k/val',
        ann_dir='labels/sem_seg/masks/val',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
        dataset_name="bdd",
        pipeline=test_pipeline)
)


data = dict(
    train=dict(
        type="CustomConcatDataset",
        datasets=[idd_data["train"], gta_data["train"], bdd_data["train"], city_data["train"], viper_data["train"], vistas_data["train"], ade_data["train"],
                  wild_data["train"], scannet_data["train"]],
    ),
    val=dict(
        type="CustomConcatDataset",
        datasets=[city_data["val"], viper_data["val"], vistas_data["val"], ade_data["val"],
                  wild_data["val"], scannet_data["val"]]
    ),
    test=city_data["test"]
)
