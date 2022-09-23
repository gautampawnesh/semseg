# Code is adapted from https://github.com/open-mmlab/mmsegmentation
import argparse
import csv
import os
import os.path as osp
import numpy as np
import shutil
import time
from time import perf_counter
import warnings
import json
import matplotlib.pyplot as plt
from matplotlib import gridspec
from PIL import Image
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.apis.inference import inference_segmentor
from mmseg.utils import get_root_logger
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes
import src.transforms
import src.datasets
import src.models
import pandas as pd

CFG_DICT = None
IMG_INFERENCE_SEED = 3  # random seed to select test images for inference
IMG_INFERENCE_NUMBER = 200  # number of images used for inference
EVALUATION_FILE = "evaluation.csv"
INF_FOLDER = "pred_labels"



def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument("-c", '--config_file_path', help='test config file path')
    parser.add_argument("-l", '--checkpoint_path', help="checkpoint file path")
    return parser.parse_args()


def config_for_evaluation(cfg):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    return cfg


def evaluate(cfg, args, logger):
    """evaluation"""
    # Distributed evaluation check
    distributed = False if cfg.launcher is None else True
    if distributed is True:
        init_dist(cfg.launcher, **cfg.get("dist_params", {}))

    # setting torch cudnn benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # build the dataloader
    dataset_name = cfg.data.test["dataset_name"]
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False,
        drop_last=False,
        seed=1,
        persistent_workers=False,
    )
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg
    # fp16_cfg = cfg.get('fp16', None)
    # if fp16_cfg is not None:
    #     wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        model.PALETTE = dataset.PALETTE

    dataset_len = len(dataset)
    if not distributed:
        mmdata_model = MMDataParallel(model, device_ids=cfg.gpu_ids[0])
        start_time = perf_counter()
        results = single_gpu_test(mmdata_model, data_loader, pre_eval=True)
        duration = (perf_counter() - start_time) / dataset_len
    else:
        dist_model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        start_time = perf_counter()
        results = multi_gpu_test(dist_model, data_loader, pre_eval=True, gpu_collect=True)
        duration = (perf_counter() - start_time) / dataset_len

    rank, world_size = get_dist_info()
    if rank == 0:
        metrics = dataset.evaluate(results, metric=["mIoU"])
        metrics["average_inference_duration"] = duration * world_size
        metrics["num_images"] = dataset_len
        logger.info(metrics)
        # # creating folder of individual dataset benchmark
        # cfg.work_dir = osp.join(cfg.work_dir, dataset_name)
        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
        # create evaluation json file for dataset
        with open(osp.join(cfg.work_dir, f"{dataset_name}_{EVALUATION_FILE}"), "w") as f:
            w = csv.writer(f)
            w.writerows(metrics.items())
        # run image inference for given dataset
        image_inference(dataset, model, cfg.work_dir)


def image_inference(dataset, model, out_dir):
    """Execute Inference on images"""
    inf_output_dir = osp.join(out_dir, f"{dataset.dataset_name}_{INF_FOLDER}")
    mmcv.mkdir_or_exist(osp.abspath(inf_output_dir))
    cls_mapping = dataset.pred_backward_mapping
    inf_results = inference_results(dataset, model)
    # POST PROCESSING: map unified classes to dataset specific classes
    inf_results = post_processing_class_mapping(cls_mapping, inf_results)
    # Label ids and class ids may be different;so convert class palette to label palette
    class_palette_to_label_palette_ind = np.zeros(max(dataset.DATASET_LABEL_IDS) + 1, dtype=int)
    for cls_id, label_id in enumerate(dataset.DATASET_LABEL_IDS):
        class_palette_to_label_palette_ind[label_id] = cls_id
    for each, result in inf_results.items():
        save_loc = inf_output_dir + "/" + f"{each}.png"
        save_inference_image(result, save_loc, dataset.DATASET_CLASSES, dataset.DATASET_PALETTE,
                             dataset.PALETTE, dataset.CLASSES, class_palette_to_label_palette_ind)


def inference_results(dataset, model):
    # random selection
    results = dict()
    dataset.data = (
        dataset.data.sort_values("image").sample(
        min(IMG_INFERENCE_NUMBER, len(dataset.data)), random_state=IMG_INFERENCE_SEED,).reset_index(drop=True)
        )
    model.eval()
    model_output = [inference_segmentor(model, img)[0] for img in dataset.data["image"]]
    for ind, row in enumerate(dataset.data.iterrows()):
        results[ind] = {
            "image": row[1]["image"],
            "label": row[1]["label"],
            "model_output": model_output[ind]
        }
    return results


def post_processing_class_mapping(dataset_uni_cls_mapping, inf_results):
    # DISCLAIMER: Only support label or train ids;
    for ind, result in inf_results.items():
        model_output = result["model_output"]

        model_output_mapped = np.zeros(model_output.shape[0:2], dtype=np.uint8)
        if len(model_output.shape) == 2:
            model_output = np.expand_dims(model_output, axis=2)
        for uni_ids, cls_id in dataset_uni_cls_mapping.items():
            for uni_id in uni_ids:
                model_output_mapped += (np.all(model_output == uni_id, axis=2).astype(dtype=np.uint8)) * cls_id

        result["processed_model_output"] = model_output_mapped
    return inf_results


def save_inference_image(result, save_loc, classes, palette, universal_palette, universal_classes, class_palette_to_label_palette_ind):
    """img+ground truth+prediction and overlay image"""
    # Todo : Update with better visualization
    # Todo: update palette for ground truth
    palette = np.array(palette)

    plt.figure(figsize=(24, 26))
    grid_spec = gridspec.GridSpec(3, 4, width_ratios=[1, 6, 6, 1])

    image = np.array(Image.open(result["image"]))
    plt.subplot(grid_spec[0, 1])
    plt.imshow(image)
    plt.axis("off")
    plt.title("Input: image")

    plt.subplot(grid_spec[1, 1])
    seg_image = palette[class_palette_to_label_palette_ind[result["processed_model_output"].astype(int)]]
    plt.imshow(seg_image)
    plt.axis("off")
    plt.title("Postprocessed Prediction: Segmentation map")

    plt.subplot(grid_spec[1, 2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis("off")
    plt.title("Postprocessed Prediction: Segmentation overlay")

    universal_palette = np.array(universal_palette).astype(np.uint8)
    plt.subplot(grid_spec[2, 1])
    uni_seg_image = universal_palette[result["model_output"].astype(int)]
    plt.imshow(uni_seg_image)
    plt.axis("off")
    plt.title("Universal Prediction: Segmentation map")

    plt.subplot(grid_spec[2, 2])
    plt.imshow(image)
    plt.imshow(uni_seg_image, alpha=0.7)
    plt.axis("off")
    plt.title("Universal Prediction: Segmentation overlay")

    if result["label"] is not None:
        label = np.array(Image.open(result["label"]))
        plt.subplot(grid_spec[0, 2])
        plt.imshow(palette[class_palette_to_label_palette_ind[label]])
        plt.axis("off")
        plt.title("Ground truth: label")

    ax = plt.subplot(grid_spec[:, 0])
    plt.imshow(np.expand_dims(universal_palette, 1))

    ax.yaxis.tick_left()
    plt.yticks(ticks=range(len(universal_classes)), labels=universal_classes)
    plt.xticks(ticks=[], labels=[])
    ax.tick_params(width=0.0)
    plt.grid("off")

    ax = plt.subplot(grid_spec[:, 3])
    plt.imshow(np.expand_dims(palette, 1))

    ax.yaxis.tick_right()
    plt.yticks(ticks=range(len(classes)), labels=classes)
    plt.xticks(ticks=[], labels=[])
    ax.tick_params(width=0.0)
    plt.grid("off")

    # Save plot
    plt.savefig(save_loc, format="png")
    plt.close()


def main():
    global CFG_DICT
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config_file_path)

    # alter config file for evaluation
    cfg = config_for_evaluation(cfg)

    # create work_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, "evaluation")
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # logger for evaluation
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config_file_path)))
    logger.info(f'Config:\n{cfg.pretty_text}')

    # GPUs
    num_gpus = torch.cuda.device_count()
    cfg["gpu_ids"] = list(range(num_gpus))

    CFG_DICT = cfg
    # Checkpoint path
    logger.info("Evaluation started ....")
    evaluate(cfg, args, logger)


if __name__ == "__main__":
    main()
