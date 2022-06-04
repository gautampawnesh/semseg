from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset

@DATASETS.register_module()
class UniversalAdeDataset(BaseDataset):

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
                 dataset_name="ade",
                 is_color_to_uni_class_mapping=False,
                 num_samples=None):
        super(UniversalAdeDataset, self).__init__(
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
