
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size = (512, 512)


train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='MapAnnotations'),
    dict(type='Resize', img_scale=(1920, 1208), ratio_range=(0.5, 2.0)),
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

cityscape_train = dict(
    type="UniversalCityscapesDataset",
    data_root="/ds-av/public_datasets/cityscapes/raw",
    img_dir='leftImg8bit/train',
    ann_dir='gtFine/train',
    seg_map_suffix='_gtFine_labelIds.png',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
    dataset_name="cityscapes",
    pipeline=train_pipeline
)

cityscape_val = dict(
    type="UniversalCityscapesDataset",
    data_root="/ds-av/public_datasets/cityscapes/raw",
    img_dir='leftImg8bit/val',
    ann_dir='gtFine/val',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
    seg_map_suffix='_gtFine_labelIds.png',
    dataset_name="cityscapes",
    pipeline=test_pipeline
)
cityscape_test = dict(
        type="UniversalCityscapesDataset",
        data_root="/ds-av/public_datasets/cityscapes/raw",
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
        seg_map_suffix='_gtFine_labelIds.png',
        dataset_name="cityscapes",
        pipeline=test_pipeline
)

viper_train = dict(
    type="UniversalViperDataset",
    data_root="/ds-av/public_datasets/viper/raw",
    img_dir='train/img',
    ann_dir='train/cls',
    seg_map_suffix='.png',
    img_suffix=".jpg",
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
    dataset_name="viper",
    pipeline=train_pipeline
)

viper_val = dict(
    type="UniversalViperDataset",
    data_root="/ds-av/public_datasets/viper/raw",
    img_dir='val/img/024',
    ann_dir='val/cls/024',
    seg_map_suffix='.png',
    img_suffix=".jpg",
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
    dataset_name="viper",
    pipeline=test_pipeline
)
viper_test = dict(
    type="UniversalViperDataset",
    data_root="/ds-av/public_datasets/viper/raw",
    img_dir='val/img',
    ann_dir='val/cls',
    seg_map_suffix='.png',
    img_suffix=".jpg",
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
    dataset_name="viper",
    pipeline=test_pipeline
)
vistas_train = dict(
        type="VistasDataset",
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='training/images',
        ann_dir='training/v1.2/labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        pipeline=train_pipeline)

vistas_val = dict(
        type="VistasDataset",
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='validation/images',
        ann_dir='validation/v1.2/labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        pipeline=test_pipeline)

vistas_test = dict(
        type="VistasDataset",
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='validation/images',
        ann_dir='validation/v1.2/labels',
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        pipeline=test_pipeline)

bdd10k_train = dict(
    type='UniversalBdd10kDataset',
    data_root='/ds-av/public_datasets/bdd100k/raw',
    img_dir='images/10k/train',
    ann_dir='labels/sem_seg/masks/train',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
    pipeline=train_pipeline)

bdd10k_val = dict(
    type='UniversalBdd10kDataset',
    data_root='/ds-av/public_datasets/bdd100k/raw',
    img_dir='images/10k/val',
    ann_dir='labels/sem_seg/masks/val',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
    pipeline=test_pipeline)

bdd10k_test = dict(
    type='UniversalBdd10kDataset',
    data_root='/ds-av/public_datasets/bdd100k/raw',
    img_dir='images/10k/val',
    ann_dir='labels/sem_seg/masks/val',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
    pipeline=test_pipeline)

idd_train = dict(
    type="UniversalIddDataset",
    data_root='/netscratch/gautam/idd_segmentation/raw',
    img_dir='leftImg8bit/train',
    ann_dir='gtFine/train',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
    pipeline=train_pipeline)

idd_val = dict(
    type="UniversalIddDataset",
    data_root='/netscratch/gautam/idd_segmentation/raw',
    img_dir='leftImg8bit/val',
    ann_dir='gtFine/val',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
    pipeline=test_pipeline)

idd_test = dict(
    type="UniversalIddDataset",
    data_root='/netscratch/gautam/idd_segmentation/raw',
    img_dir='leftImg8bit/val',
    ann_dir='gtFine/val',
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
    pipeline=test_pipeline)

playing_for_data_train = dict(
    type="UniversalPlayingForDataDataset",
    data_root="/ds-av/public_datasets/playing_for_data/raw",
    img_dir='images',
    ann_dir='labels',
    split="/netscratch/gautam/playing_for_data/train.txt",
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/playing_for_data_class_mapping.csv",
    pipeline=train_pipeline)

playing_for_data_val = dict(
    type="UniversalPlayingForDataDataset",
    data_root="/ds-av/public_datasets/playing_for_data/raw",
    img_dir='images',
    ann_dir='labels',
    split="/netscratch/gautam/playing_for_data/val.txt",
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/playing_for_data_class_mapping.csv",
    pipeline=test_pipeline)

playing_for_data_test = dict(
    type="UniversalPlayingForDataDataset",
    data_root="/ds-av/public_datasets/playing_for_data/raw",
    img_dir='images',
    ann_dir='labels',
    split="/netscratch/gautam/playing_for_data/val.txt",
    universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
    dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/playing_for_data_class_mapping.csv",
    pipeline=test_pipeline)

data = dict(
    train=dict(
        type="CustomConcatDataset",
        datasets=[cityscape_train, viper_train, bdd10k_train, idd_train, playing_for_data_train, vistas_train],
    ),
    val=dict(
        type="CustomConcatDataset",
        datasets=[cityscape_val, vistas_val]
    ),
    test=viper_test
    # test=dict(
    #     type="CustomConcatDataset",
    #     datasets=[cityscape_test]#, viper_test, bdd10k_test, idd_test, vistas_test]
    # )
)
