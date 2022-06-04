import pandas as pd
import numpy as np
import seaborn
import mmcv
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines.compose import Compose
from mmseg.datasets.pipelines.loading import LoadAnnotations
from src.transforms.annotations import MapAnnotations
from pathlib import Path
import logging
import os.path as osp
from collections import OrderedDict
from mmcv.utils import print_log
from prettytable import PrettyTable
from mmseg.core import pre_eval_to_metrics, intersect_and_union

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
                # self.split = osp.join(self.data_root, self.split)
                # Todo: playing for data doesnt have split folder. and split files are at diff location
                self.split = self.split

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
            self.cls_mapping = self.dataset_ids_to_universal_label_mapping()
        self.DATASET_CLASSES, self.DATASET_PALETTE = self.set_dataset_classes_and_palette()
        self.pred_backward_mapping = self.set_pred_backward_class_mapping()
        self.data = self.data_df()

    def set_dataset_classes_and_palette(self):
        """
        dataset original classes and palette,
        These are used for evaluation and visualization of dataset data
        :return:
        """
        class_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        dataset_classes = tuple(class_mapping_df["dataset_class_name"].tolist())
        dataset_palette = [list(map(int, str(color).split(","))) for color in class_mapping_df["dataset_color_code"]]
        return dataset_classes, dataset_palette

    def set_pred_backward_class_mapping(self):
        """
        set mapping for predictions from Unified (universal) classes to dataset specific classes.
        :return:
        """
        pred_class_mapping = dict()
        class_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        dataset_label_ids = class_mapping_df["dataset_label_id"].tolist()
        universal_data_ids = class_mapping_df["universal_to_dataset"].tolist()
        for uni_ids, cls_id in zip(universal_data_ids, dataset_label_ids):
            if pd.isna(uni_ids):
                continue
            pred_class_mapping[tuple(map(int, uni_ids.split(",")))] = int(cls_id)
        return pred_class_mapping

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

        self.PALETTE = ((np.array(seaborn.color_palette("Set1", len(self.CLASSES)))*255).astype(np.uint8).tolist())
        self.num_classes = len(self.CLASSES)

    def dataset_colors_to_universal_label_mapping(self):
        """
        Maps a dataset class color codes to universal class indexes.
        :return:
        """

        dataset_cls_mapping_df = pd.read_csv(self.dataset_class_mapping_path, delimiter=";")
        color_tuples = [list(map(int, str(color).split(","))) for color in dataset_cls_mapping_df["dataset_color_code"]]
        # Todo: if colors are in bgr convert to rgb
        color_tuples = [tuple(color) for color in color_tuples]
        uni_cls_ids = dataset_cls_mapping_df["universal_class_id"].tolist()
        mapping = dict(zip(color_tuples, uni_cls_ids))
        return mapping

    def images_labels_validation(self, images):
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
                self.img_dir, self.ann_dir).replace(self.img_suffix, self.seg_map_suffix))
            # VIPER dataset has some empty images
            if img_path.stat().st_size != 0 and label_path.stat().st_size != 0:
                # Todo: playing for data has white images
                valid_images.append(img_path)
                valid_labels.append(label_path)
                # try:
                #     if osp.basename(img_path).replace(self.img_suffix, self.seg_map_suffix) == osp.basename(label_path):
                #         valid_images.append(img_path)
                #         valid_labels.append(label_path)
                #     else:
                #         raise ValueError
                # except Exception as err:
                #     err.args += (img_path, label_path)
                #     raise

        return valid_images, valid_labels

    def data_df(self):
        """data df with image path and annotations"""
        images = list(Path(self.img_dir).glob(f"**/*{self.img_suffix}"))
        images, labels = self.images_labels_validation(images)
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
        if not self.test_mode:
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

    def pred_backward_class_mapping(self, preds):
        """
        Args:
            preds (list[torch.Tensor] | torch.Tensor): the segmentation logit
                after argmax, shape (N, H, W).
        :param preds:
        :return:
        """
        # Todo: convert to tensor
        preds_mapped = []
        for pred in preds:
            pred_mapped = np.zeros(pred.shape[0:2])
            if len(pred.shape) == 2:
                pred = np.expand_dims(pred, axis=2)
            for uni_ids, cls_id in self.pred_backward_mapping.items():
                for uni_id in uni_ids:
                    pred_mapped += (np.all(pred == uni_id, axis=2))*cls_id
            preds_mapped.append(pred_mapped)
        return preds_mapped

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

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=logger,
                 gt_seg_maps=None,
                 **kwargs):
        """Evaluate the dataset.

        Args:
            results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
                 results or predict segmentation map for computing evaluation
                 metric.
            metric (str | list[str]): Metrics to be evaluated. 'mIoU',
                'mDice' and 'mFscore' are supported.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            gt_seg_maps (generator[ndarray]): Custom gt seg maps as input,
                used in ConcatDataset

        Returns:
            dict[str, float]: Default metrics.
        """
        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError('metric {} is not supported'.format(metric))

        eval_results = {}
        # test a list of files
        if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
                results, str):
            raise NotImplementedError
        # test a list of pre_eval_results
        else:
            ret_metrics = pre_eval_to_metrics(results, metric)

        # Because dataset.CLASSES is required for per-eval.
        if self.DATASET_CLASSES is None:
            raise NotImplementedError
            #class_names = tuple(range(num_classes))
        else:
            class_names = self.DATASET_CLASSES

        # summary table
        ret_metrics_summary = OrderedDict({
            ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })

        # each class table
        ret_metrics.pop('aAcc', None)
        ret_metrics_class = OrderedDict({
            ret_metric: np.round(ret_metric_value * 100, 2)
            for ret_metric, ret_metric_value in ret_metrics.items()
        })
        ret_metrics_class.update({'Class': class_names})
        ret_metrics_class.move_to_end('Class', last=False)

        # for logger
        class_table_data = PrettyTable()
        for key, val in ret_metrics_class.items():
            class_table_data.add_column(key, val)

        summary_table_data = PrettyTable()
        for key, val in ret_metrics_summary.items():
            if key == 'aAcc':
                summary_table_data.add_column(key, [val])
            else:
                summary_table_data.add_column('m' + key, [val])

        print_log('per class results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        print_log('Summary:', logger)
        print_log('\n' + summary_table_data.get_string(), logger=logger)

        # each metric dict
        for key, value in ret_metrics_summary.items():
            if key == 'aAcc':
                eval_results[key] = value / 100.0
            else:
                eval_results['m' + key] = value / 100.0

        ret_metrics_class.pop('Class', None)
        for key, value in ret_metrics_class.items():
            eval_results.update({
                key + '.' + str(name): value[idx] / 100.0
                for idx, name in enumerate(class_names)
            })

        return eval_results

    def __getitem__(self, index):

        results = self.pre_pipeline(index)
        return self.pipeline(results)

    def __len__(self):
        """length of the dataset."""
        return len(self.data)
