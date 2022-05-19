import torch
import time
import copy
import argparse
from mmcv.runner import init_dist
from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmcv import Config


def train(cfg, launcher):
    """training """
    num_gpus = torch.cuda.device_count()
    cfg["gpu_ids"] = list(range(num_gpus))
    distributed = False if launcher is None else True
    if distributed is True:
        init_dist(launcher, **cfg.get("dist_params", {}))
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())

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
    # Build the detector
    model = build_segmentor(cfg.model)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
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
    cfg = Config.fromfile(args.config_file_path)
    train(cfg, "slurm")
