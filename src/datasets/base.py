import pandas as pd
import numpy as np
import seaborn
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines.compose import Compose
from mmseg.datasets.pipelines.loading import LoadAnnotations
from src.transforms.annotations import MapAnnotations
from pathlib import Path
import logging
import os.path as osp

logger = logging.getLogger(__name__)


class BaseDataset(CustomDataset):
    """
    Base Dataset class for hierarchical semseg network
    Args:
        pipeline: procressing pipeline
        img_dir: Path to img directory
        img_suffix: suffix of images
        ann_dir: path to annotation direc
        seg_map_suffix: suffix of segmentation maps
        split: split txt file, ** if split is specified, only file with suffix in the splits
            will be loaded. Otherwise, all images in img_dir/ann_dir will be loaded.
        data_root: data root for img_dir/ann_dir
        test_mode(bool): if test_mode=True, gt wouldn't be loaded.
        ignore_index(int): The label index to be ignored, default :255
        reduce_zero_label: Whether to mark label zero as ignored/
        classes: Specify classes to load. **if is None, cls.CLASSES will be used.
        palette: The palette of segmentation map. if None is given, and self.PALETTE is None, random palette will be
            generated.
        gt_seg_map_loader_cfg: build LoadAnnotations to load gt for evaluation, load from disk by default.
        file_client_args: arguments to instatiate a FileClient.
    """
    CLASSES = None
    PALETTE = None

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix=".jpg",
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=False,
                 ignore_index=255,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend="disk"),
                 class_color_mode="RGB",
                 universal_class_colors_path=None,
                 dataset_class_mapping=None,
                 dataset_name="base",
                 is_color_to_uni_class_mapping=True):
        self.pipeline = Compose(pipeline)
        self.img_dir = img_dir
        self.img_suffix = img_suffix
        self.ann_dir = ann_dir
        self.seg_map_suffix = seg_map_suffix
        self.split = split
        self.data_root = data_root
        self.test_mode = test_mode
        self.ignore_index = ignore_index
        self.reduce_zero_label = reduce_zero_label
        self.label_map = {}
        self.dataset_name = dataset_name
        self.is_color_to_uni_class_mapping = is_color_to_uni_class_mapping
        # join paths if data_root is specified
        if self.data_root is not None:
            if not osp.isabs(self.img_dir):
                self.img_dir = osp.join(self.data_root, self.img_dir)
            if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
                self.ann_dir = osp.join(self.data_root, self.ann_dir)
            if not (self.split is None or osp.isabs(self.split)):
                self.split = osp.join(self.data_root, self.split)

        self.universal_class_colors_path = Path(universal_class_colors_path) if universal_class_colors_path else None
        self.dataset_class_mapping_path = Path(dataset_class_mapping) if dataset_class_mapping else None
        assert self.universal_class_colors_path is not None, "Universal class colors path is missing"
        assert self.dataset_class_mapping_path is not None, "Dataset class colors path is missing"
        self.load_annotations = LoadAnnotations()
        self.map_annotations = MapAnnotations()
        # define cls.CLASSES and cls.PALETTE
        self.set_universal_classes_and_palette()

        if self.is_color_to_uni_class_mapping:
            self.cls_mapping = self.dataset_colors_to_universal_label_mapping()
        else:
            self.cls_mapping = self.dataset_ids_to_universal_label_mapping()  ## should be implemented by subclasses
        self.data = self.data_df()

    def dataset_ids_to_universal_label_mapping(self):
        raise NotImplementedError

    def set_universal_classes_and_palette(self):
        """
        Set cls.CLASSES and cls.PALETTE
        :return:
        """
        universal_color_df = pd.read_csv(self.universal_class_colors_path, delimiter=";")
        self.CLASSES = universal_color_df["class_name"][universal_color_df["class_name"].notnull()].tolist()
        logger.info(f"Universal classes list: {self.CLASSES}")

        self.PALETTE = ((np.array(seaborn.color_palette("tab20", len(self.CLASSES)))*255).astype(np.uint8).tolist())
        self.num_classes = len(self.CLASSES)

    def dataset_colors_to_universal_label_mapping(self):
        """
        Maps a dataset class color codes to universal class indexes.
        :return:
        """

        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        color_tuples = [list(map(int, str(color).split(","))) for color in dataset_cls_mapping_df["dataset_color_code"]]
        # Todo: if colors are in bgr convert to rgb
        logger.info("CHECKPOINT: Color code format !!!! should be RGB")
        color_tuples = [tuple(color) for color in color_tuples]
        logger.info(f"Dataset color tuples: {color_tuples}")
        uni_cls_ids = dataset_cls_mapping_df["universal_class_id"].tolist()
        logger.info(f"Dataset color_codes len: {len(color_tuples)}, uni_cls_ids len: {len(uni_cls_ids)}")
        mapping = dict(zip(color_tuples, uni_cls_ids))
        logger.info(f"Mapped to universal class ids: {mapping}")
        return mapping

    def data_df(self):
        """data df with image path and annotations"""
        images = list(Path(self.img_dir).glob(f"**/*{self.img_suffix}"))
        labels = list(Path(self.ann_dir).glob(f"**/*{self.seg_map_suffix}"))
        images = sorted(images)
        labels = sorted(labels)
        logger.info("CHECKPOINT: Is images and labels indexes are correct ??")
        logger.info(f"{self.dataset_name} Images: {images[:5]} \n | Labels: {labels[:5]} ")
        logger.info(f"Images len: {len(images)}, Labels: {len(labels)}")
        logger.info("CHECKPOINT: Labels might be missing for certain images !!!!!!!!!!")
        data_df = pd.DataFrame.from_dict({"image": images, "label": labels})
        return data_df.sort_values("image")

    def pre_pipeline(self, ind):
        """returns sample wo processing"""
        data = self.data.iloc[ind]
        results = {
            "mapping": self.cls_mapping,
            "num_classes": self.num_classes,
            "img_info": {"filename": data["image"]},
            "ann_info": {"seg_map": data["label"]},
            "seg_fields": []
        }
        return results

    def get_gt_seg_map_by_idx(self, index):
        results = self.pre_pipeline(index)
        results = self.load_annotations(results)
        results = self.map_annotations(results)
        return results["gt_semantic_seg"]

    def get_gt_seg_maps(self, efficient_test=None):
        if efficient_test is not None:
            logger.warning(
                "DeprecationWarning: ``efficient_test`` has been deprecated "
                "since MMSeg v0.16, the ``get_gt_seg_maps()`` is CPU memory "
                "friendly by default. "
            )
        for ind in range(len(self)):
            yield self.get_gt_seg_map_by_idx(ind)

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 gt_seg_maps=None,
                 **kwargs):
        evaluation_results = CustomDataset.evaluate(self, results, metric='mIoU', logger=logger, gt_seg_maps=gt_seg_maps, **kwargs)
        #logger.info(evaluation_results)
        return evaluation_results

    def __getitem__(self, index):

        results = self.pre_pipeline(index)
        return self.pipeline(results)

    def __len__(self) -> int:
        """length of the dataset."""
        return len(self.data)
