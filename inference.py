import argparse
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
from tqdm import tqdm



def config_for_evaluation(cfg):
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True
    cfg.model.train_cfg = None
    return cfg


def inference_results(dataset, model):
    # random selection
    results = dict()
    model.eval()
    model.cuda()
    model_output = [inference_segmentor(model, img)[0] for img in tqdm(dataset.data["image"])]
    for ind, row in enumerate(dataset.data.iterrows()):
        results[ind] = {
            "image": row[1]["image"],
            "model_output": model_output[ind]

        }
    return results

def save_inference_image(result, save_loc, classes, palette, universal_palette, universal_classes, class_palette_to_label_palette_ind):
    palette = np.array(palette)
    try:
        seg_image = result["processed_model_output"].astype(np.uint8)
    except Exception as e:
        raise

    im_out = Image.fromarray(seg_image)

    # im = Image.open(result["image"])
    # arr_im = np.array(im)
    # print(f"input size: {arr_im.shape} output_siye {seg_image.shape}")


    im_out.save(f"{save_loc}")
    # plt.imshow(seg_image)
    # # Save plot
    # plt.savefig(save_loc, format="png")
    # plt.close()


def image_inference(dataset, model, out_dir):
    """Execute Inference on images"""
    inf_output_dir = osp.join(out_dir, f"{dataset.dataset_name}")
    mmcv.mkdir_or_exist(osp.abspath(inf_output_dir))
    cls_mapping = dataset.pred_backward_mapping
    inf_results = inference_results(dataset, model)
    # POST PROCESSING: map unified classes to dataset specific classes
    inf_results = post_processing_class_mapping(cls_mapping, inf_results)
    # Label ids and class ids may be different;so convert class palette to label palette
    class_palette_to_label_palette_ind = np.zeros(max(dataset.DATASET_LABEL_IDS) + 1, dtype=int)
    for cls_id, label_id in enumerate(dataset.DATASET_LABEL_IDS):
        class_palette_to_label_palette_ind[label_id] = cls_id
    for each, result in tqdm(inf_results.items()):
        #out_name = str(result["image"]).split(".")[0].split("/")[-1]
        # Cityscape
        #out_name = result["image"].stem.split("_leftImg8bit")[0]
        # Ade+wilddash
        out_name = result["image"].stem
        save_loc = inf_output_dir + "/" + f"{out_name}.png"
        save_inference_image(result, save_loc, dataset.DATASET_CLASSES, dataset.DATASET_PALETTE,
                             dataset.PALETTE, dataset.CLASSES, class_palette_to_label_palette_ind)


def post_processing_class_mapping(dataset_uni_cls_mapping, inf_results):
    # DISCLAIMER: Only support label or train ids;
    for ind, result in tqdm(inf_results.items()):
        model_output = result["model_output"]

        model_output_mapped = np.zeros(model_output.shape[0:2], dtype=np.uint8)
        if len(model_output.shape) == 2:
            model_output = np.expand_dims(model_output, axis=2)
        for uni_ids, cls_id in dataset_uni_cls_mapping.items():
            for uni_id in uni_ids:
                model_output_mapped += (np.all(model_output == uni_id, axis=2).astype(dtype=np.uint8)) * cls_id

        result["processed_model_output"] = model_output_mapped
    print("post processing completed.")
    return inf_results


def inference_start(cfg, args, logger):
    """Benchmark Inference """
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
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=distributed,
    #     shuffle=False,
    #     drop_last=False,
    #     seed=1,
    #     persistent_workers=False,
    # )
    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.cfg = cfg

    checkpoint = load_checkpoint(model, args.checkpoint_path, map_location='cpu')
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        model.CLASSES = dataset.CLASSES

    image_inference(dataset, model, cfg.work_dir)


def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument("-c", '--config_file_path', help='test config file path')
    parser.add_argument("-l", '--checkpoint_path', help="checkpoint file path")
    return parser.parse_args()


def main():
    global CFG_DICT
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config_file_path)

    # alter config file for evaluation
    cfg = config_for_evaluation(cfg)

    # create work_dir
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    cfg.work_dir = osp.join(cfg.work_dir, "inference")
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
    logger.info("Inference started ....")
    inference_start(cfg, args, logger)


if __name__ == "__main__":
    main()


