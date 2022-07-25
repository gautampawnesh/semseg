import multiprocessing
from pathlib import Path
from mmcv.utils.config import Config
from mmseg.datasets import build_dataset
from torch.utils.data import DataLoader
import src.datasets
import src.transforms
import numpy as np
from tqdm import tqdm
import mmcv
from mmcv.parallel import MMDistributedDataParallel, MMDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint)
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.models import build_segmentor
from mmseg.datasets import build_dataloader, build_dataset

import pandas as pd
import src.datasets
import src.transforms
import src.models
import torch.nn as nn
import torch
import cv2
from PIL import Image
from collections import defaultdict

import matplotlib.pyplot as plt

NUM_DATASETS=6
NUM_SAMPLES_PER_DATASET=10
checkpoint_path = "/netscratch/gautam/semseg/exp_results/FMD7/training/20220705_062238/epoch_90.pth"
NUM_CLASSES = 191
config_file_path = "./configs/inference/test_FMD7_r101-d8_512x512_all.py"
classes_df = pd.read_csv("./configs/_base_/class_mapping/universal_classes.csv", sep=";")

CLASSES= {}
for key, value in zip(classes_df["class_id"], classes_df["class_name"]):
    CLASSES[key] = value


class ProjectionHead(nn.Module):
    def __init__(self, dim_in, norm_cfg=None, proj_dim=256, proj='linear'):
        super(ProjectionHead, self).__init__()

        if proj == 'linear':
            self.proj = nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        else:
            pass

    def forward(self, x):
        return torch.nn.functional.normalize(self.proj(x), p=2, dim=1)

proj_head = ProjectionHead(dim_in=2048, proj="linear", proj_dim=256)

cfg = mmcv.Config.fromfile(config_file_path)
cfg.model.pretrained = None
cfg.data.test.test_mode = True
cfg.model.train_cfg = None
distributed = False if cfg.launcher is None else True
if distributed is True:
    init_dist(cfg.launcher, **cfg.get("dist_params", {}))

dataset = build_dataset(cfg.data.val)
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
model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
model.cfg = cfg
checkpoint = load_checkpoint(model, checkpoint_path, map_location='cpu')
num_gpus = torch.cuda.device_count()
cfg["gpu_ids"] = list(range(num_gpus))

results=None
dataset_len = len(dataset)
if distributed:
    dist_model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False)
    results = multi_gpu_test(dist_model, data_loader, pre_eval=False, gpu_collect=False, feature=True)

if not distributed:
    mmdata_model = MMDataParallel(model, device_ids=cfg.gpu_ids)
    results = single_gpu_test(mmdata_model, data_loader, pre_eval=False, feature=True)

proj_results = proj_head(torch.stack(results).to("cpu"))
proj_results = proj_results.detach()

input_img, label = [], []
for data in data_loader:
    input_img.append(data["img"].data[0])
    label.append(data["gt_semantic_seg"].data[0].to(torch.uint8))

p_labels = []
feature_w, feature_h = 64, 64
for i_label in label:
    label_arr = i_label.squeeze(dim=0).cpu().detach().numpy()
    p_labels.append(torch.from_numpy(cv2.resize(label_arr.squeeze(), dsize=(64, 64), interpolation=cv2.INTER_NEAREST)))

pixel_embedding = dict()

for proj_res, label in zip(proj_results, p_labels):
    assert proj_res.shape[-2:] == label.shape[-2:]

    # repeat along dim 1
    processed_label = label.unsqueeze(dim=0).repeat(256, 1, 1)

    for each_cls in range(NUM_CLASSES):
        # print(proj_res[processed_label==each_cls].reshape(-1, 256).shape)
        cls_embedding = proj_res[processed_label == each_cls].reshape(-1, 256)
        # try:
        if pixel_embedding.get(each_cls) is not None and cls_embedding.shape[0] != 0:
            # print("tesn")
            pixel_embedding[each_cls] = torch.cat((pixel_embedding[each_cls], cls_embedding), 0)
        elif cls_embedding.shape[0] != 0:
            # import pdb; pdb.set_trace()
            pixel_embedding[each_cls] = cls_embedding

        # print(each_cls)
    # keys = list(pixel_embedding.keys())
    # print(keys)

keys = list(pixel_embedding.keys())

df_dict = {}
for i_key in sorted(keys):
    df_dict[i_key] = torch.mean(pixel_embedding[i_key], dim=0).detach().cpu().numpy()

feat_mean_df = pd.DataFrame.from_dict(df_dict)
feat_mean_df.to_csv(f"feat_{NUM_SAMPLES_PER_DATASET}.csv", index=False)