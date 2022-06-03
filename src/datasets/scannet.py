from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset
import pandas as pd
from pathlib import Path
import json
import os.path as osp

@DATASETS.register_module()
class UniversalScannetDataset(BaseDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix=".jpg",
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=None,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend="disk"),
                 class_color_mode="RGB",
                 universal_class_colors_path=None,
                 dataset_class_mapping=None,
                 dataset_name="scannet",
                 is_color_to_uni_class_mapping=True,
                 num_samples=None):
        super(UniversalScannetDataset, self).__init__(
            pipeline,
            img_dir,
            img_suffix=img_suffix,
            ann_dir=ann_dir,
            seg_map_suffix=seg_map_suffix,
            split=split,
            data_root=data_root,
            test_mode=test_mode,
            ignore_index=ignore_index,
            reduce_zero_label=reduce_zero_label,
            classes=classes,
            palette=palette,
            gt_seg_map_loader_cfg=gt_seg_map_loader_cfg,
            file_client_args=file_client_args,
            class_color_mode=class_color_mode,
            universal_class_colors_path=universal_class_colors_path,
            dataset_class_mapping=dataset_class_mapping,
            dataset_name=dataset_name,
            is_color_to_uni_class_mapping=is_color_to_uni_class_mapping)
        self.num_samples = num_samples

    def data_df(self):
        """fetch data from the disk"""
        if self.split:
            images, labels = [], []
            with open(self.split, "r") as fp:
                data = json.load(fp)
            for each_img, each_ann in zip(data["images"], data["annotations"]):
                images.append(Path(osp.join(self.img_dir, each_img['file_name'])))
                labels.append(Path(osp.join(self.ann_dir, each_ann['file_name'])))
            data_df = pd.DataFrame.from_dict({"image": images, "label": labels})
            data_df = data_df.sort_values("image")
            return data_df if self.samples is None else data_df.sample(n=self.num_samples)
        else:
            raise NotImplementedError
