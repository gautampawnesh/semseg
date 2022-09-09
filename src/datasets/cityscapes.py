# gt has Label Ids: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py

from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset
import pandas as pd


@DATASETS.register_module()
class UniversalCityscapesDataset(BaseDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix="_leftImg8bit.png",
                 ann_dir=None,
                 seg_map_suffix='_gtFine_labelIds.png',
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
                 dataset_name="cityscapes",
                 is_color_to_uni_class_mapping=False,
                 num_samples=None,
                 data_seed=1,
                 benchmark=False,
                 extra_img_dir=None,
                 extra_ann_dir=None
                 ):
        # mark all non eval classes to 0 based on gt label id
        self.gt_non_eval_classes = [1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        super(UniversalCityscapesDataset, self).__init__(
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
            benchmark=benchmark,
            extra_img_dir=extra_img_dir,
            extra_ann_dir=extra_ann_dir
        )

    def dataset_ids_to_universal_label_mapping(self):
        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        label_ids = dataset_cls_mapping_df["dataset_label_id"].tolist()
        label_ids = [(id,) for id in label_ids]
        uni_cls_ids = dataset_cls_mapping_df["universal_class_id"].tolist()
        mapping = dict(zip(label_ids, uni_cls_ids))
        return mapping
