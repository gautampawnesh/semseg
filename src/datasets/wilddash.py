from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset
import pandas as pd
from pathlib import Path
import json
import os.path as osp

@DATASETS.register_module()
class UniversalWilddashDataset(BaseDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix=".jpg",
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=None,
                 ignore_index=0,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend="disk"),
                 class_color_mode=None,
                 universal_class_colors_path=None,
                 dataset_class_mapping=None,
                 dataset_name="wilddash",
                 is_color_to_uni_class_mapping=False,
                 num_samples=None):

        # mark all non eval classes to 0 based on gt label id
        self.gt_non_eval_classes = [2, 3, 4, 5, 6, 9, 10, 15, 16, 29, 30, 31]
        super(UniversalWilddashDataset, self).__init__(
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
            is_color_to_uni_class_mapping=is_color_to_uni_class_mapping,
            num_samples=num_samples
        )

    def data_df(self):
        """fetch data from the disk"""
        if self.split:
            raise NotImplementedError
        images = list(Path(self.img_dir).glob(f"**/*{self.img_suffix}"))
        if self.test_mode:
            images = images[3556:]
        else:
            images = images[:3556]
        if self.num_samples:
            import random
            random.seed(1)
            images = random.sample(images, self.num_samples)
        images, labels = self.images_labels_validation(images)
        data_df = pd.DataFrame.from_dict({"image": images, "label": labels})
        data_df = data_df.sort_values("image")
        return data_df

    def dataset_ids_to_universal_label_mapping(self):
        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        label_ids = dataset_cls_mapping_df["dataset_label_id"].tolist()
        label_ids = [(id,) for id in label_ids]
        uni_cls_ids = dataset_cls_mapping_df["universal_class_id"].tolist()
        mapping = dict(zip(label_ids, uni_cls_ids))
        return mapping
