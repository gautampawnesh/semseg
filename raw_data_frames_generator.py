# ugly Configuration
train_pipeline, val_pipeline, test_pipeline = [], [], []
data_seed = 0
city_data = dict(
    train=dict(
        type="UniversalCityscapesDataset",
        data_root='/ds-av/public_datasets/cityscapes/raw',
        img_dir='leftImg8bit/train',
        img_suffix="_leftImg8bit.png",
        ann_dir='gtFine/train',
        seg_map_suffix='_gtFine_labelIds.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
        dataset_name="cityscapes"),
    val=dict(
        type="UniversalCityscapesDataset",
        data_root='/ds-av/public_datasets/cityscapes/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        img_suffix="_leftImg8bit.png",
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
        img_dir='leftImg8bit/test',
        img_suffix="_leftImg8bit.png",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/cityscapes_class_mapping.csv",
        seg_map_suffix='_gtFine_labelIds.png',
        ignore_index=0,  # gt has Labelids
        dataset_name="cityscapes",
        benchmark=True,
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
        ignore_index=0,
        test_mode=True,
        pipeline=val_pipeline),
    test=dict(
        type="UniversalViperDataset",
        data_root='/ds-av/public_datasets/viper/raw',
        img_dir='test/img',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/viper_class_mapping.csv",
        dataset_name="viper",
        ignore_index=0,  # gt are Label ids
        benchmark=True,
        pipeline=test_pipeline))

vistas_data = dict(
    train=dict(
        type='VistasDataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='training/images',
        ann_dir='training/v1.2/labels',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='VistasDataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='validation/images',
        ann_dir='validation/v1.2/labels',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        test_mode=True,
        benchmark=True,
        pipeline=test_pipeline),
    test=dict(
        type='VistasDataset',
        data_root='/ds-av/public_datasets/mapillary_vistas_v2.0/raw/',
        img_dir='testing/images',
        seg_map_suffix='.png',
        class_color_mode="RGB",
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/vistas_class_mapping.csv",
        dataset_name="vistas",
        benchmark=True,
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
        img_suffix=".jpg",
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalAdeDataset',
        data_root='/ds-av/public_datasets/ade_20k/raw',
        img_dir='images/validation',
        img_suffix=".jpg",
        ann_dir='annotations/validation',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/ade20k_class_mapping.csv",
        seg_map_suffix='.png',
        is_color_to_uni_class_mapping=False,
        dataset_name="ade",
        test_mode=True,
        pipeline=val_pipeline),
    test=dict(
        type='UniversalAdeDataset',
        data_root='/netscratch/gautam/ade_test',
        img_dir='release_test/testing',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/ade20k_class_mapping.csv",
        seg_map_suffix='.png',
        is_color_to_uni_class_mapping=False,
        dataset_name="ade",
        benchmark=True,
        pipeline=test_pipeline))

wild_data = dict(
    train=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        seg_map_suffix='.png',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        is_color_to_uni_class_mapping=False,
        dataset_name="wilddash",
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/public',
        img_dir='images',
        ann_dir='labels',
        img_suffix=".jpg",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        seg_map_suffix='.png',
        is_color_to_uni_class_mapping=False,
        test_mode=True,
        dataset_name="wilddash",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalWilddashDataset',
        data_root='/ds-av/public_datasets/wilddash2/raw/benchmark',
        img_dir='images',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/wilddash_class_mapping.csv",
        seg_map_suffix='.png',
        img_suffix=".jpg",
        is_color_to_uni_class_mapping=False,
        dataset_name="wilddash",
        benchmark=True,
        pipeline=test_pipeline))

scannet_data = dict(
    train=dict(
        type='UniversalScannetDataset',
        data_root='/netscratch/gautam/scannet/',
        img_dir='scans',
        ann_dir='scans',
        seg_map_suffix='_labelId.png',
        img_suffix=".jpg",
        split="/netscratch/gautam/scannet/scannet_train.json",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/scannet_class_mapping.csv",
        is_color_to_uni_class_mapping=False,
        dataset_name="scannet",
        data_seed=data_seed,
        pipeline=train_pipeline),
    val=dict(
        type='UniversalScannetDataset',
        data_root='/netscratch/gautam/scannet/',
        img_dir='scans',
        ann_dir='scans',
        img_suffix=".jpg",
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
        img_suffix=".jpg",
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
        seg_map_suffix='_gtFine_labelids.png',
        img_suffix="_leftImg8bit.png",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
        pipeline=train_pipeline),
    val=dict(
        type='UniversalIddDataset',
        data_root='/netscratch/gautam/idd_segmentation/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        img_suffix="_leftImg8bit.png",
        seg_map_suffix='_gtFine_labelids.png',
        dataset_name="idd",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/idd_class_mapping.csv",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalIddDataset',
        data_root='/netscratch/gautam/idd_segmentation/raw',
        img_dir='leftImg8bit/val',
        ann_dir='gtFine/val',
        img_suffix="_leftImg8bit.png",
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
        img_suffix=".png",
        seg_map_suffix='.png',
        split="/netscratch/gautam/playing_for_data/train.txt",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/playing_for_data_class_mapping.csv",
        pipeline=train_pipeline),
    val=dict(
        type='UniversalPlayingForDataDataset',
        data_root='/ds-av/public_datasets/playing_for_data/raw',
        img_dir='images',
        img_suffix=".png",
        ann_dir='labels',
        seg_map_suffix='.png',
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
        img_suffix=".png",
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
        img_suffix=".jpg",
        seg_map_suffix='.png',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
        pipeline=train_pipeline),
    val=dict(
        type='UniversalBdd10kDataset',
        data_root='/ds-av/public_datasets/bdd100k/raw',
        img_dir='images/10k/val',
        ann_dir='labels/sem_seg/masks/val',
        img_suffix=".jpg",
        seg_map_suffix='.png',
        dataset_name="bdd",
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
        pipeline=val_pipeline),
    test=dict(
        type='UniversalBdd10kDataset',
        data_root='/ds-av/public_datasets/bdd100k/raw',
        img_dir='images/10k/val',
        img_suffix=".jpg",
        ann_dir='labels/sem_seg/masks/val',
        universal_class_colors_path="/netscratch/gautam/semseg/configs/_base_/class_mapping/universal_classes.csv",
        dataset_class_mapping="/netscratch/gautam/semseg/configs/_base_/class_mapping/bdd10k_class_mapping.csv",
        dataset_name="bdd",
        pipeline=test_pipeline)
)
from multiprocessing import Pool
from functools import partial
import multiprocessing
from pathlib import Path
from mmcv.utils.config import Config
from mmseg.datasets import build_dataset
from torch.utils.data import DataLoader
import src.datasets
import src.transforms
import numpy as np
from tqdm import tqdm
import pandas as pd
import os.path as osp


def images_labels_validation(images, img_dir, ann_dir, img_suffix, seg_map_suffix):
    """
    Args:
         images: list of images
    """
    # workaround for invalid images Todo
    pre_defined_invalid_images = [
        Path('/ds-av/public_datasets/playing_for_data/raw/images/15188.png'),
        Path('/ds-av/public_datasets/playing_for_data/raw/images/17705.png')
    ]
    valid_images, valid_labels = [], []
    for img_path in images:
        if img_path in pre_defined_invalid_images:
            continue
        label_path = Path(str(img_path).replace(
            img_dir, ann_dir).replace(img_suffix, seg_map_suffix))
        # VIPER dataset has some empty images
        if img_path.stat().st_size != 0 and label_path.stat().st_size != 0:
            # Todo: playing for data has white images
            valid_images.append(img_path)
            valid_labels.append(label_path)
    data_list = [{"image": img, "label": label} for img, label in zip(valid_images, valid_labels)]
    data_df = pd.DataFrame.from_records(data_list)
    return data_df


def process_single_dataset(data_type, data):
    img_suffix = data[data_type]["img_suffix"]
    seg_map_suffix = data[data_type]["seg_map_suffix"]
    img_dir = osp.join(data[data_type]["data_root"], data[data_type]["img_dir"])
    ann_dir = osp.join(data[data_type]["data_root"], data[data_type]["ann_dir"])

    images = list(Path(img_dir).glob(f"**/*{img_suffix}"))
    data_df = images_labels_validation(images, img_dir, ann_dir, img_suffix, seg_map_suffix)
    return data_df

datasets = [idd_data, gta_data, bdd_data]
WORKERS = len(datasets)
data_type="train"
print("Started ..")

with Pool(WORKERS) as p:
    results = p.map(partial(process_single_dataset, data_type), datasets)

for result_df, data in zip(results,  datasets):
    dataset_name = data[data_type]["dataset_name"]
    result_df.to_csv(f"./raw_data_metadata/{dataset_name}_raw_{data_type}_dataset.csv", index=False)
