# Code is adapted from https://github.com/AutoNUE/public-code
# JSON to LabelImage
from anue_labels import name2label
from annotation import Annotation
import os
import sys
import glob
from argparse import ArgumentParser
from tqdm import tqdm
import numpy
args = None
# Image processing
# Check if PIL is actually Pillow as expected
# try:
#     from PIL import PILLOW_VERSION
# except:
#     print("Please install the module 'Pillow' for image processing, e.g.")
#     print("pip install pillow")
#     sys.exit(-1)

try:
    import PIL.Image as Image
    import PIL.ImageDraw as ImageDraw
except:
    print("Failed to import the image processing packages.")
    sys.exit(-1)

# Convert the given annotation to a label image


def createLabelImage(inJson, annotation, encoding, outline=None):
    # the size of the image
    size = (annotation.imgWidth, annotation.imgHeight)

    # the background
    if encoding == "id":
        background = name2label['unlabeled'].id
    elif encoding == "csId":
        background = name2label['unlabeled'].csId
    elif encoding == "csTrainId":
        background = name2label['unlabeled'].csTrainId
    elif encoding == "level4Id":
        background = name2label['unlabeled'].level4Id
    elif encoding == "level3Id":
        background = name2label['unlabeled'].level3Id
    elif encoding == "level2Id":
        background = name2label['unlabeled'].level2Id
    elif encoding == "level1Id":
        background = name2label['unlabeled'].level1Id
    elif encoding == "color":
        background = name2label['unlabeled'].color
    else:
        print("Unknown encoding '{}'".format(encoding))
        return None

    # this is the image that we want to create
    if encoding == "color":
        labelImg = Image.new("RGBA", size, background)
    else:
        # print(size, background)
        labelImg = Image.new("L", size, background)

    # a drawer to draw into the image
    drawer = ImageDraw.Draw(labelImg)

    # loop over all objects
    for obj in annotation.objects:
        label = obj.label
        polygon = obj.polygon
        val = None

        # If the object is deleted, skip it
        if obj.deleted or len(polygon) < 3:
            continue

        # If the label is not known, but ends with a 'group' (e.g. cargroup)
        # try to remove the s and see if that works
        if (not label in name2label) and label.endswith('group'):
            label = label[:-len('group')]

        if not label in name2label:
            print("Label '{}' not known.".format(label))
            tqdm.write("Something wrong in: " + inJson)
            continue

        # If the ID is negative that polygon should not be drawn
        if name2label[label].id < 0:
            continue

        if encoding == "id":
            val = name2label[label].id
        elif encoding == "csId":
            val = name2label[label].csId
        elif encoding == "csTrainId":
            val = name2label[label].csTrainId
        elif encoding == "level4Id":
            val = name2label[label].level4Id
        elif encoding == "level3Id":
            val = name2label[label].level3Id
        elif encoding == "level2Id":
            val = name2label[label].level2Id
        elif encoding == "level1Id":
            val = name2label[label].level1Id
        elif encoding == "color":
            val = name2label[label].color

        try:
            if outline:

                drawer.polygon(polygon, fill=val, outline=outline)
            else:
                drawer.polygon(polygon, fill=val)
                # print(label, val)
        except:
            print("Failed to draw polygon with label {}".format(label))
            raise

    # print(numpy.array(labelImg))

    return labelImg


def json2labelImg(inJson, outImg, encoding="ids"):
    annotation = Annotation()
    annotation.fromJsonFile(inJson)
    labelImg = createLabelImage(inJson, annotation, encoding)
    labelImg.save(outImg)


def process_folder(fn):
    global args

    dst = fn.replace("_polygons.json", "_label{}s.png".format(args.id_type))

    # do the conversion
    try:
        json2labelImg(fn, dst, args.id_type)
    except:
        tqdm.write("Failed to convert: {}".format(fn))
        raise


def main(args):
    import sys
    # how to search for all ground truth
    searchFine = os.path.join(args.datadir, "gtFine",
                              "*", "*", "*_gt*_polygons.json")
    print(searchFine)
    # search files
    filesFine = glob.glob(searchFine)
    filesFine.sort()
    print(filesFine)

    files = filesFine  # filesFine
    if not files:
        print("Did not find any files.")
    tqdm.write(
        "Processing {} annotation files for Sematic Segmentation".format(len(files)))
    # iterate through files
    progress = 0
    try:
        tqdm.write("Progress: {:>3} %".format(
            progress * 100 / len(files)), end=' ')
    except ZeroDivisionError:
        print("no data")

    from multiprocessing import Pool
    import time

    pool = Pool(args.num_workers)
    # results = pool.map(process_pred_gt_pair, pairs)
    results = list(
        tqdm(pool.imap(process_folder, files), total=len(files)))
    pool.close()
    pool.join()


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--datadir', default="/netscratch/gautam/idd_segmentation/raw")
    parser.add_argument('--id-type', default='id')
    parser.add_argument('--num-workers', type=int, default=10)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    main(args)
