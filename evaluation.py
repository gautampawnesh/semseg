# Code is adapted from https://github.com/open-mmlab/mmsegmentation
import argparse
import os
import os.path as osp
import shutil
import time
from time import perf_counter
import warnings
import json

import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from mmseg import digit_version
from mmseg.utils import get_root_logger
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import setup_multi_processes

IMG_INFERENCE_SEED = 1  # random seed to select test images for inference
IMG_INFERENCE_NUMBER = 20  # number of images used for inference
EVALUATION_FILE = "evaluation.json"
INF_FOLDER = "visualizations"


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
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
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
        model = MMDataParallel(model, device_ids=cfg.gpu_ids)
        start_time = perf_counter()
        results = single_gpu_test(model, data_loader, pre_eval=True)
        duration = (perf_counter() - start_time) / dataset_len
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        start_time = perf_counter()
        results = multi_gpu_test(model, data_loader, pre_eval=True, gpu_collect=True)
        duration = (perf_counter() - start_time) / dataset_len

    rank, world_size = get_dist_info()
    if rank == 0:
        metrics = dataset.evaluate(results, ["mIoU"])
        metrics["average_inference_duration"] = duration * world_size
        metrics["num_images"] = dataset_len
        logger.info(metrics)

        with open(osp.join(cfg.work_dir, EVALUATION_FILE), "w") as f:
            json.dump(metrics, f, indent=4)


def image_inference(cfg, dataset, model, out_dir):
    """Execute Inference on images"""
    inf_output_dir = osp.join(out_dir, INF_FOLDER)
    mmcv.mkdir_or_exist(osp.abspath(inf_output_dir))
    inf_results = inference_results(dataset, model)


def inference_results(dataset, model):
    # random selection
    dataset.sample_df = ()
    pass



def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config_file_path)

    # alter config file for evaluation
    cfg = config_for_evaluation(cfg)

    # create work_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, "evaluation", timestamp)
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

    # Checkpoint path
    # Todo: set checkpoint default path to best_mIoU checkpoint.

    logger.info("Evaluation started ....")
    evaluate(cfg, args, logger)


if __name__ == "__main__":
    main()
