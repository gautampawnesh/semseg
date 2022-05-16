import streamlit as st
import argparse
def main(class_color_path: str,
         image_path: str,
         label_path: str):

    pass


# if "_name_" == '_main_':
    # use all available space and make image path entirely readable
    #st.set_page_config(layout="wide")
    # Set title of app
st.title('Semantic Segmentation Dataset Viewer')
    # Set title for sidebar
st.sidebar.title("Settings")

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-i", "--image_dir", required=False, help="path to image dir, can also be changed in frontend",
    #                     )
    # parser.add_argument("-l", "--label_dir", required=False, help="path to label dir, can also be changed in frontend",
    #                     )
    # parser.add_argument("-c", "--class_color_path", required=False,
    #                     help="path to class color csv, can also be changed in frontend",
    #                     )
    # args = parser.parse_args()
    # run main
    # main(class_color_path=args.class_color_path,
    #      image_path=args.image_dir,
    #      label_path=args.label_dir)