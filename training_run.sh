# srun -K --partition=A100 --nodes=1 --ntasks=2 --cpus-per-task=10 --gpus-per-task=1 --container-image=/netscratch/gautam/mmlab_dlcc_semseg_20.0_1-py3.sqsh --container-workdir="`pwd`" --container-mounts="`pwd`":"`pwd`",/ds-av:/ds-av:ro --mem-per-cpu=6G sh training_run.sh
# srun -K --partition=A100 --nodes=1 --ntasks=4 --cpus-per-task=10 --gpus-per-task=1 --container-image=/netscratch/gautam/mmlab_22.01_1.sqsh --container-workdir="`pwd`" --container-mounts="`pwd`":"`pwd`",/ds-av:/ds-av:ro --mem-per-cpu=6G sh training_run.sh

#git clone https://github.com/open-mmlab/mmcv.git
#cd mmcv
#MMCV_WITH_OPS=1 pip install -e .
#cd ..
#git clone https://github.com/open-mmlab/mmsegmentation.git
#cd mmsegmentation
#pip install -e .
#cd ..
#apt-get install -y libgl1-mesa-glxu
#pip install opencv-python
#pip install mmcv-full==1.5.0
#pip uninstall -y mmsegmentation==0.24.1
#pip install cityscapesScripts==2.2.0
#pip install seaborn
python training.py -c /netscratch/gautam/semseg/configs/slurm/deeplabv3plus_r50-d8_512x1024_vistas.py
