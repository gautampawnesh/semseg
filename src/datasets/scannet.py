import torch

from mmseg.datasets.builder import DATASETS
from src.datasets.base import BaseDataset
import pandas as pd
from pathlib import Path
import json
import os.path as osp
from mmseg.core import intersect_and_union


@DATASETS.register_module()
class UniversalScannetDataset(BaseDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix=".jpg",
                 ann_dir=None,
                 seg_map_suffix='_labelId.png',
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
                 dataset_name="scannet",
                 is_color_to_uni_class_mapping=True,
                 num_samples=None,
                 data_seed=1,
                 benchmark=False
                 ):

        # mark all non eval classes to 0 based on gt label id
        self.gt_non_eval_classes = []
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
            is_color_to_uni_class_mapping=is_color_to_uni_class_mapping,
            num_samples=num_samples,
            data_seed=data_seed,
            benchmark=benchmark)

    def dataset_ids_to_universal_label_mapping(self):
        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        label_ids = dataset_cls_mapping_df["dataset_label_id"].tolist()
        label_ids = [(id,) for id in label_ids]
        uni_cls_ids = dataset_cls_mapping_df["universal_class_id"].tolist()
        mapping = dict(zip(label_ids, uni_cls_ids))
        return mapping

    def data_df(self):
        """fetch data from the disk"""
        if self.split:
            images, labels = [], []
            with open(self.split, "r") as fp:
                data = json.load(fp)
            for each_img, each_ann in zip(data["images"], data["annotations"]):
                ann_file = each_img['file_name'].replace("frame/color", "annotation/segmentation")
                ann_file = ann_file.replace(self.img_suffix, self.seg_map_suffix)
                images.append(Path(osp.join(self.img_dir, each_img['file_name'])))
                labels.append(Path(osp.join(self.ann_dir, ann_file)))
            data_df = pd.DataFrame.from_dict({"image": images, "label": labels})
            data_df = data_df.sort_values("image")
            if self.num_samples:
                try:
                    data_df = data_df.sample(n=self.num_samples, random_state=self.data_seed)
                except:
                    data_df = data_df.sample(n=self.num_samples, replace=True, random_state=self.data_seed)

            return data_df
        else:
            raise NotImplementedError

    def pre_eval(self, preds, indices):
        """
        Dataset specific evaluation, ground truth and prediction are converted to dataset specific classes means
        It maps universal classes to dataset classes (backward mapping)

        Collect eval result from each iteration.

        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
            indices (list[int] | int): the prediction related ground truth
                indices.

        Returns:
            list[torch.Tensor]: (area_intersect, area_union, area_prediction,
                area_ground_truth).
        """
        # In order to compat with batch inference
        if not isinstance(indices, list):
            indices = [indices]
        if not isinstance(preds, list):
            preds = [preds]
        # universal classes to dataset classes mapping
        preds = self.pred_backward_class_mapping(preds)

        pre_eval_results = []

        for pred, index in zip(preds, indices):
            # In test mode, seg_map will receive dataset specific labels
            seg_map = self.get_gt_seg_map_by_idx(index)
            # Todo: Remove this
            import cv2
            try:
                seg_map = cv2.resize(seg_map, dsize=(pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
            except Exception as e:
                e.args += (torch.from_numpy(seg_map).shape, pred.shape)
                raise
            pre_eval_results.append(
                intersect_and_union(
                    pred,
                    seg_map,
                    len(self.DATASET_CLASSES),
                    self.ignore_index,
                    # as the labels has been converted when dataset initialized
                    # in `get_palette_for_custom_classes ` this `label_map`
                    # should be `dict()`, see
                    # https://github.com/open-mmlab/mmsegmentation/issues/1415
                    # for more ditails
                    label_map=dict(),
                    reduce_zero_label=self.reduce_zero_label))

        return pre_eval_results
