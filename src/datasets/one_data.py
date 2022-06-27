from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
from src.datasets.cityscapes import UniversalCityscapesDataset
from src.datasets.ade import UniversalAdeDataset
from src.datasets.bdd10k import UniversalBdd10kDataset
from src.datasets.idd import UniversalIddDataset
from src.datasets.playing_for_data import UniversalPlayingForDataDataset
from src.datasets.scannet import UniversalScannetDataset
from src.datasets.viper import UniversalViperDataset
from src.datasets.vistas import VistasDataset
from src.datasets.wilddash import UniversalWilddashDataset
import pandas as pd


@DATASETS.register_module()
class OneDataset(CustomDataset):

    def __init__(self,
                 pipeline,
                 img_dir,
                 img_suffix=".jpg",
                 ann_dir=None,
                 seg_map_suffix='.png',
                 split=None,
                 data_root=None,
                 test_mode=None,
                 ignore_index=0,
                 reduce_zero_label=False,
                 classes=None,
                 palette=None,
                 gt_seg_map_loader_cfg=None,
                 file_client_args=dict(backend="disk"),
                 class_color_mode=None,
                 universal_class_colors_path=None,
                 dataset_class_mapping=None,
                 dataset_name="ade",
                 is_color_to_uni_class_mapping=False,
                 num_samples=None):
        #self.pipeline = Compose(pipeline)
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


        # parameters for each dataset class is different.
        #for each_dataset in []:
            # make an instance of it class
            # list of instances

