import logging
from pathlib import Path
import numpy as np
from mmcv.utils import print_log
from mmseg.datasets.builder import PIPELINES

logger = logging.getLogger(__name__)


@PIPELINES.register_module()
class MapAnnotations(object):
    """Map annotations to integer class ID."""

    def __call__(self, results):
        dataset_uni_cls_mapping = results.get("mapping")
        gt_semantic_seg = results.get("gt_semantic_seg")
        gt_semantic_seg_mapped = np.zeros(gt_semantic_seg.shape[0:2], dtype=np.uint8)
        if len(gt_semantic_seg.shape) == 2:
            gt_semantic_seg = np.expand_dims(gt_semantic_seg, axis=2)
        for label_id, universal_ind in dataset_uni_cls_mapping.items():
            gt_semantic_seg_mapped += (np.all(gt_semantic_seg == label_id, axis=2).astype(dtype=np.uint8)) * universal_ind

        # logger.info("CHECKPOINT: MapAnnotations !!!")
        # if len(gt_semantic_seg.shape) == 2:
        #     gt_semantic_seg = np.expand_dims(gt_semantic_seg, axis=2)
        # gt_semantic_seg_mapped = np.zeros(gt_semantic_seg.shape[0:2], dtype=np.uint8)
        # logger.info(f"gt_semantic_seg shape: {gt_semantic_seg.shape}. !!")
        # logger.info("gt_shape len shouldn't be 2 and 3 channels")
        # for color_tup, universal_ind in dataset_uni_cls_mapping.items():
        #     gt_semantic_seg_mapped += (
        #         np.all(gt_semantic_seg == color_tup, axis=2).astype(dtype=np.uint8)
        #     ) * universal_ind

        results["gt_semantic_seg"] = gt_semantic_seg_mapped

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__
