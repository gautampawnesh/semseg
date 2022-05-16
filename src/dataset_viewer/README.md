# Dataset-viewer
This repository contains code for visualizing semantic segmentated images and labels.

#### Setup Python environment

pip install -r requirements.txt


#### Run the tool

streamlit run app.py  --server.address 0.00.0 --server.port 8501 -- -i /images/ -l /labels/ -c /dataset_color.csv

streamlit run app.py  --server.address 0.00.0 --server.port 8501
