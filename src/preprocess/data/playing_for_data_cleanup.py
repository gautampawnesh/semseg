from tqdm import tqdm
import glob
import PIL.Image as Image
from multiprocessing import Pool
import numpy as np
from os import path as osp

DATA_DIR = "/ds-av/public_datasets/playing_for_data/raw/images"
OUTPUT_DIR = "/netscratch/gautam/"
NUM_WORKERS = 20
invalid_files = []


def process_image(img_path):
    pic = Image.open(img_path)
    pic_arr = np.array(pic)
    if np.all(pic_arr == 0) or np.all(pic_arr == 255):
        invalid_files.append(str(img_path))


def write_data(data_list):
    np.savetxt(f"{OUTPUT_DIR}playing_for_data_invalid_files.txt", np.array(invalid_files), fmt="%s", delimiter=",")


def main():
    searchImg = osp.join(DATA_DIR, "*.png")
    files = glob.glob(searchImg)
    if not files:
        print("Did not find any files.")
    tqdm.write(
        "Processing {} files for cleaning".format(len(files)))

    # iterate through files
    progress = 0
    try:
        tqdm.write("Progress: {:>3} %".format(
            progress * 100 / len(files)), end=' ')
    except ZeroDivisionError:
        print("no data")

    pool = Pool(NUM_WORKERS)
    results = list(
        tqdm(pool.imap(process_image, files), total=len(files)))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
    write_data(invalid_files)