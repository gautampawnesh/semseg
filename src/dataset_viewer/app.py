import os
import argparse
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image

BASE_PATH = os.getcwd()


print(os.environ)

def rgbstr_to_rgb(rgbstr):
    rgb = [int(color) for color in rgbstr.split(",")]
    return tuple(rgb)


def rgb_to_rgbstr(r, g, b):
    return "{},{},{}".format(r, g, b)


def rgb2hex(rgbstr):
    rgb = [int(color) for color in rgbstr.split(",")]
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def get_all_images(path, data_type):
    paths = []
    for (dirpath, _, filenames) in os.walk(path, topdown=False):
        for filename in filenames:
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
                if data_type == "label" and args.dataset_name.lower() == "cityscape":
                    if filename.endswith("gtFine_color.png"):
                        temp = os.path.join(dirpath, filename)
                        paths.append(temp)
                elif data_type == "image" and args.dataset_name.lower() == "viper":
                    if filename.endswith(".jpg"):
                        temp = os.path.join(dirpath, filename)
                        paths.append(temp)
                else:
                    temp = os.path.join(dirpath, filename)
                    paths.append(temp)
    paths.sort()
    return paths

def create_data_sidebar(class_color_path,
                        image_path,
                        label_path):
    # Reading class color CSV files
    st.sidebar.header("Data")
    class_color_path = st.sidebar.text_input('Enter path to class color specification:', value=class_color_path)
    try:
        class_color_df = pd.read_csv(class_color_path)
    except FileNotFoundError as error_msg:
        st.sidebar.error(error_msg)

    try:
        # Check if needed columns are available
        class_colors = class_color_df["ColorCode"]
        class_names = class_color_df["ClassName"]
        class_color_dict = {}
        for class_color, class_name in zip(class_colors, class_names):
            class_color_dict[class_color] = class_name
    except KeyError:
        st.sidebar.error("ClassColor or ClassName is not available in CSV File")

    # Reading SemSeg images and labels
    image_path = st.sidebar.text_input('Enter path to images:', value=image_path)
    label_path = st.sidebar.text_input('Enter path to labels:', value=label_path)

    # Get all paths
    image_paths = get_all_images(image_path, "image")
    label_paths = get_all_images(label_path, "label")
    st.header("There are {} images and {} labels".format(
        len(image_paths), len(label_paths)))
    if len(image_paths) != len(label_paths):
        st.sidebar.error("There should be the same amount of images as labels".format(
            len(image_paths), len(label_paths)))
    return class_color_dict, (image_paths, label_paths)

def create_class_color_checkboxes(class_color_dict):
    # Read CSV file
    st.sidebar.header("Classes")
    slider = st.sidebar.slider(label="Transparency", min_value=0.0, value=0.4, max_value=1.0, step=0.1)

    # put all needed information in a list
    class_checkboxes = []
    st.sidebar.text('\nWhich classes do you want to see?')
    for class_color in class_color_dict.keys():
        # create checkbox with color
        color = st.sidebar.markdown("""
            <svg width="80" height="20">
            <rect width="80" height="20" style="fill:{};stroke-width:3;stroke:rgb(0,0,0)" />
            </svg>""".format(rgb2hex(class_color)), unsafe_allow_html=True)
        box = st.sidebar.checkbox("{}".format(class_color_dict[class_color]))

        # append everything to list
        class_checkboxes.append((box, class_color, class_color_dict[class_color]))
    return class_checkboxes, slider

@st.cache(suppress_st_warning=True)
def create_images_per_class(paths):
    classes_per_image = []
    for image_path, label_path in zip(*paths):
        label = Image.open(label_path)
        colors = label.convert('RGB').getcolors()

        class_per_image = {}
        for _, color in colors:
            key = rgb_to_rgbstr(*color)
            class_per_image[key] = 1
        class_per_image["ImagePath"] = image_path
        class_per_image["LabelPath"] = label_path
        classes_per_image.append(class_per_image)

    classes_per_image_df = pd.DataFrame(classes_per_image)
    return classes_per_image_df

@st.cache(suppress_st_warning=True)
def create_overlay_image(image_path, label_path, alpha):
    image = Image.open(image_path)
    label = Image.open(label_path)

    image = image.convert("RGBA")
    label = label.convert("RGBA")

    overlay = Image.blend(image, label, alpha)
    st.image(overlay, use_column_width=True)
    return overlay


def get_files_from_dir(base_dir, extension=None, use_str=None):
    """
    Get files with pre-defined extensions from directory and sub-dirs.
    Folders can be excluded or included.
    Args:
        base_dir (str): Root directory
        extensions (strings): File extensions which are of interest.
        use_str (strings): Name of folder which will be included.
    Returns:
        Full path of all the files.
    """
    res = []
    for path, name, files in os.walk(base_dir):
        for f in files:
            if (extension is None or f.endswith(extension)) and (use_str is None or path[-len(use_str):] == use_str):
                res.append(os.path.join(path, f))
    return res


def main(class_color_path: str,
         image_path: str,
         label_path: str):
    class_color_dict, paths = create_data_sidebar(class_color_path, image_path, label_path)
    class_checkboxes, slider = create_class_color_checkboxes(class_color_dict)
    print("paths")
    print(paths)
    classes_per_image_df = create_images_per_class(paths)

    # get all checked boxes
    marked_boxes = []
    for box, class_color, class_name in class_checkboxes:
        if box:
            marked_boxes.append(class_color)
    print(classes_per_image_df)
    marked = classes_per_image_df[marked_boxes + ["ImagePath", "LabelPath"]].dropna()

    image_list = marked["ImagePath"].tolist()
    label_list = marked["LabelPath"].tolist()
    selected_image = st.selectbox(
        "Which image do you want to see?",
        image_list
    )
    create_overlay_image(selected_image, label_list[image_list.index(selected_image)], slider)


# use all available space and make image path entirely readable
st.set_page_config(layout="wide")
# Set title of app
st.title('Semantic Segmentation Dataset Viewer')
# Set title for sidebar
st.sidebar.title("Settings")

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset_name", help="Dataset name")
parser.add_argument("-i", "--image_dir", help="path to image dir, can also be changed in frontend")
parser.add_argument("-l", "--label_dir",  help="path to label dir, can also be changed in frontend")
parser.add_argument("-c", "--class_color_path",
                    help="path to class color csv, can also be changed in frontend")
args = parser.parse_args()
print(args.image_dir)
print(args.label_dir)
# run main
main(class_color_path=args.class_color_path,
     image_path=args.image_dir,
     label_path=args.label_dir)
