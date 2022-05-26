# Code is adapted from https://github.com/open-mmlab/mmsegmentation
import torch
import time
import copy
import argparse
import mmcv
import os.path as osp
from mmseg import __version__
from mmcv.runner import init_dist
from mmcv.utils import get_git_hash
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import get_root_logger
from mmcv import Config
import src.transforms
import src.datasets


def train(config_file_path: str):
    """
    training based on OpenMMlab
    """
    cfg = Config.fromfile(config_file_path)
    # gpus
    num_gpus = torch.cuda.device_count()
    cfg["gpu_ids"] = list(range(num_gpus))

    # Distributed Training
    distributed = False if cfg.get("launcher") is None else True
    if distributed is True:
        init_dist(cfg.get("launcher"), **cfg.get("dist_params", {}))

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    # create work_dir
    if cfg.resume_from is None:
        cfg.work_dir = osp.join(cfg.work_dir, "training", timestamp)
    else:
        cfg.work_dir = osp.dirname(cfg.resume_from)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(config_file_path)))

    # init the logger before other steps
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    meta = {"exp_name": cfg.get("experiment", {}).get("name")}

    if cfg.get("seed") is not None:
        meta["seed"] = cfg.get("seed")
        set_random_seed(cfg.get("seed"), deterministic=False)

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))

    # Build the segmentor
    model = build_segmentor(cfg.model)
    logger.info(model)

    if cfg.checkpoint_config is not None:
        # save mmseg version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES

    # passing checkpoint meta for saving best checkpoint
    meta.update(cfg.checkpoint_config.meta)

    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        meta=meta,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_file_path", help="training config file path")

    args = parser.parse_args()
    train(args.config_file_path)
