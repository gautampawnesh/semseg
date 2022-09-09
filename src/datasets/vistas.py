from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset
import pandas as pd
from pathlib import Path


@DATASETS.register_module()
class VistasDataset(BaseDataset):

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
                 dataset_name="vistas",
                 is_color_to_uni_class_mapping=False,
                 num_samples=None,
                 data_seed=1,
                 is_extra_class_mapping=False,
                 extra_class_map=None,
                 benchmark=False
                 ):
        # mark all non eval classes to 0 based on gt label id
        self.gt_non_eval_classes = []
        super(VistasDataset, self).__init__(
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
            is_extra_class_mapping=is_extra_class_mapping,
            extra_class_map=extra_class_map,
            benchmark=benchmark
        )

    def dataset_ids_to_universal_label_mapping(self):
        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        label_ids = dataset_cls_mapping_df["dataset_class_id"].tolist()
        label_ids = [(id,) for id in label_ids]
        uni_cls_ids = dataset_cls_mapping_df["universal_class_id"].tolist()
        mapping = dict(zip(label_ids, uni_cls_ids))
        return mapping

    def data_df(self):
        """fetch data from the disk"""
        if self.split:
            raise NotImplementedError
        images = list(Path(self.img_dir).glob(f"**/*{self.img_suffix}"))
        if not self.benchmark:
            images, labels = self.images_labels_validation(images)
            data_df = pd.DataFrame.from_dict({"image": images, "label": labels})
            data_df = data_df.sort_values("image")
        else:
            data_df = pd.DataFrame.from_dict({"image": images})
        if self.num_samples is None:
            return data_df
        else:
            try:
                return data_df.sample(n=self.num_samples, random_state=self.data_seed)
            except Exception as e:
                return data_df.sample(n=self.num_samples, replace=True, random_state=self.data_seed)
