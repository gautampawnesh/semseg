from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset
import pandas as pd
from pathlib import Path


@DATASETS.register_module()
class Vistas2Dataset(BaseDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix=".jpg",
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=None,
                 ignore_index=123,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend="disk"),
                 class_color_mode="RGB",
                 universal_class_colors_path=None,
                 dataset_class_mapping=None,
                 dataset_name="vistas2",
                 is_color_to_uni_class_mapping=False,
                 num_samples=None,
                 data_seed=1,
                 is_extra_class_mapping=False,
                 extra_class_map=None,
                 img_meta_data=None,
                 benchmark=False,
                 is_universal_network=True
                 ):
        # mark all non eval classes to 0 based on gt label id
        self.gt_non_eval_classes = [2, 31, 38, 79, 86, 96, 115, 123]
        super(Vistas2Dataset, self).__init__(
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
            img_meta_data=img_meta_data,
            benchmark=benchmark,
            is_universal_network=is_universal_network
        )

    def dataset_ids_to_universal_label_mapping(self):
        """Forward Mapping"""
        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        # column should be dataset_label_id
        label_ids = dataset_cls_mapping_df["dataset_label_id"].tolist()
        label_ids = [(id,) for id in label_ids]
        uni_cls_ids = dataset_cls_mapping_df[self.forward_mapping_column].tolist()
        mapping = dict(zip(label_ids, uni_cls_ids))
        return mapping

    def data_df(self):
        """fetch data from the disk"""
        if self.split:
            raise NotImplementedError

        if self.benchmark:
            images = list(Path(self.img_dir).glob(f"**/*{self.img_suffix}"))
            data_df = pd.DataFrame.from_dict({"image": images})
        elif self.img_meta_data:
            data_df = pd.read_csv(self.img_meta_data)
            data_df = data_df.sort_values("image")
        else:
            images = list(Path(self.img_dir).glob(f"**/*{self.img_suffix}"))
            images, labels = self.images_labels_validation(images)
            data_df = pd.DataFrame.from_dict({"image": images, "label": labels})
            data_df = data_df.sort_values("image")

        if self.num_samples is None:
            return data_df
        else:
            try:
                return data_df.sample(n=self.num_samples, random_state=self.data_seed)
            except Exception as e:
                return data_df.sample(n=self.num_samples, replace=True, random_state=self.data_seed)