# Run command on slurm
# srun -K --ntasks=1 --cpus-per-task=2 --mem-per-cpu=16GB --container-image=/netscratch/enroot/dlcc_pytorch_20.07.sqsh --container-workdir="`pwd`" --container-mounts=/netscratch/$USER:/netscratch/$USER,/ds-av:/ds-av:ro,"`pwd`":"`pwd`" sh run.sh

pip install streamlit
pip install environs
# vista dataset view
streamlit run /netscratch/gautam/semseg/src/dataset_viewer/app.py  --server.address 0.00.0 --server.port 8501 -- -d mapillary -i /ds-av/public_datasets/mapillary_vistas_v2.0/raw/validation/images/ -l /ds-av/public_datasets/mapillary_vistas_v2.0/raw/validation/v1.2/labels/ -c /netscratch/gautam/semseg/src/dataset_viewer/vistas_color_codes.csv

# cityscape
# streamlit run /netscratch/gautam/semseg/src/dataset_viewer/app.py  --server.address 0.00.0 --server.port 8501 -- -d cityscape -i /ds-av/public_datasets/cityscapes/raw/leftImg8bit/train/aachen/ -l /ds-av/public_datasets/cityscapes/raw/gtFine/train/aachen/ -c /netscratch/gautam/semseg/src/dataset_viewer/cityscape_color_codes.csv

# camvid
# streamlit run /netscratch/gautam/semseg/src/dataset_viewer/app.py  --server.address 0.00.0 --server.port 8501 -- -d camvid -i /ds-av/public_datasets/camvid/raw/Camvid/701_StillsRaw_full -l /ds-av/public_datasets/camvid/raw/Camvid/LabeledApproved_full -c /netscratch/gautam/semseg/src/dataset_viewer/camvid_color_codes.csv

# viper
#streamlit run /netscratch/gautam/semseg/src/dataset_viewer/app.py  --server.address 0.00.0 --server.port 8501 -- -d viper -i /ds-av/public_datasets/viper/raw/train/img/076/ -l /ds-av/public_datasets/viper/raw/train/cls/076/ -c /netscratch/gautam/semseg/src/dataset_viewer/viper_color_codes.csv
