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

        if results.get("is_extra_class_mapping") is True:
            extra_class_mapping = results.get("extra_class_map")
            if extra_class_mapping:
                gt_semantic_seg_mapped_2 = np.zeros(gt_semantic_seg_mapped.shape[0:2], dtype=np.uint8)
                gt_semantic_seg_mapped = np.expand_dims(gt_semantic_seg_mapped, axis=2)
                for universal_key, universal_values in extra_class_mapping.items():
                    for universal_value in universal_values:
                        gt_semantic_seg_mapped_2 += (np.all(gt_semantic_seg_mapped == universal_value, axis=2).astype(
                            dtype=np.uint8)) * universal_key
                gt_semantic_seg_mapped = gt_semantic_seg_mapped_2

        results["gt_semantic_seg"] = gt_semantic_seg_mapped

        return results

    def __repr__(self) -> str:
        return self.__class__.__name__
