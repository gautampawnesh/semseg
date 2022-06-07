from PIL import Image
import numpy as np
import glob
import argparse
import os.path as osp
from collections import defaultdict


def main(args):
    uniq_labels_dict = defaultdict(int)
    uniq_labels, counter = [], []
    dir = args.dir
    files = glob.glob(osp.join(dir, f"*{args.suffix}"))
    for each_file in files:
        try:
            pic_arr = np.array(Image.open(each_file))
            uniq_labels, counter = np.unique(pic_arr, return_counts=True)
        except:
            print(f"error: opening file {each_file}")

        for uniq_label, cnt in zip(uniq_labels, counter):
            uniq_labels_dict[uniq_label] += cnt
    return uniq_labels_dict


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("-d", "--dir", help="input directory")
    parse.add_argument("-s", "--suffix", default=".png")
    args = parse.parse_args()
    unq_labels = main(args)
    print(f"unique_labels: {unq_labels.keys()}")
    print(f"unique_labels_with count: {unq_labels}")
