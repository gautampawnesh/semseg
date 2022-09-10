# TrainIds mapping are taken from: https://github.com/bdd100k/bdd100k/blob/master/bdd100k/label/label.py

from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset
import pandas as pd


@DATASETS.register_module()
class UniversalBdd10kDataset(BaseDataset):

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
                 class_color_mode=None,
                 universal_class_colors_path=None,
                 dataset_class_mapping=None,
                 dataset_name="bdd10k",
                 is_color_to_uni_class_mapping=False,
                 num_samples=None,
                 data_seed=1,
                 img_meta_data=None,
                 benchmark=False
                ):
        super(UniversalBdd10kDataset, self).__init__(
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
            num_samples=num_samples,
            data_seed=data_seed,
            img_meta_data=img_meta_data,
            benchmark=benchmark
        )

    def dataset_ids_to_universal_label_mapping(self):
        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        label_ids = dataset_cls_mapping_df["dataset_train_id"].tolist()
        label_ids = [(id,) for id in label_ids]
        uni_cls_ids = dataset_cls_mapping_df["universal_class_id"].tolist()
        mapping = dict(zip(label_ids, uni_cls_ids))
        return mapping

    def set_dataset_classes_and_palette(self):
        return [], [], []

    def set_pred_backward_class_mapping(self):
        return {}
