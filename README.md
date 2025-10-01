# MMFSeg

The official pytorch implementation of **MMFSeg: Multi-structure Multi-feature Fusion for Segmentation of Road Potholes**. ([TASE](https://ieeexplore.ieee.org/document/11176062/))


We test our code in Python 3.8, CUDA 11.3, cuDNN 8, and PyTorch 1.12.1. We provide `Dockerfile` to build the docker image we used. You can modify the `Dockerfile` as you want.  
<div align=center>
<img src="https://github.com/lab-sun/MMFSeg/blob/main/img/overall.jpg" width="900px"/>
</div>

# Results

We conducted tests on both the Pothole-600 dataset and the NO4K dataset.
A visualisation of the test results on the Pothole-600 dataset is shown below:
<div align=center>
<img src="https://github.com/lab-sun/MMFSeg/blob/main/img/Pothole-600-results.jpg" width="900px"/>
</div>

A visualisation of the test results on the NO-4K dataset is shown below:
<div align=center>
<img src="https://github.com/lab-sun/MMFSeg/blob/main/img/NO4K-result.jpg" width="900px"/>
</div>



# Introduction

MMFSeg with an RGB-Disparity fusion network that adopts CNN and Transformer as dual encoders and is equipped with a late-fusion Multi-feature Alignment Fusion (MAF) module for the segmentation of road potholes in autonomous driving environments.

# Datasets

The **Pothole-600** dataset can be downloaded from [here](https://nas.labsun.org/downloads/2025_tase_mmfseg/Pothole-600.zip). The **NO-4K** dataset can be downloaded from [here](https://nas.labsun.org/downloads/2025_tase_mmfseg/)


# Pretrained weights
The pretrained weights of MMFSeg for the two datasets can be downloaded from [here](https://nas.labsun.org/downloads/2025_tase_mmfseg/).

# Usage
* Clone this repo
```
$ git clone https://github.com/lab-sun/MMFSeg.git
```
* Build docker image
```
$ cd ~/MMFSeg
$ docker build -t docker_image_MMFSeg .
```
* Download the dataset. If you want to use NO-4K dataset, you can replace the 'Pothole' with 'NO4K' in the code below.
```
$ (You should be in the MMFSeg folder)
$ mkdir ./Pothole
$ cd ./Pothole
$ (download our preprocessed Pothole-600.zip in this folder)
$ unzip -d . Pothole-600.zip
```

* To reproduce our results, you need to download our pretrained weights. 
```
$ (You should be in the MMFSeg folder)
$ mkdir ./weights_backup
$ cd ./weights_backup
$ (download our preprocessed weights_backup.zip in this folder)
$ unzip -d . weights_backup.zip
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_MMFSeg --gpus all -v ~/MMFSeg:/workspace docker_image_MMFSeg
$ (currently, you should be in the docker)
$ cd /workspace
$ (To reproduce the results)
$ python3 run_demo_C2T_Conv0_Trans0_Pothole.py #If you want to reproduce the results of NO-4K datatset, please use python3 run_demo_C2T_Conv0_Trans0_NO4K.py
```
The results will be saved in the `./C2T_Conv0_Trans0_Pothole` folder. 

* To train MMFSeg.

During training, MMFSet first loads SegFormer's pre-trained weights, so you must first download these pre-trained weight files. They can be downloaded from [here](https://nas.labsun.org/downloads/2025_tase_mmfseg/).

```
$ cd ~/MMFSeg
$ (download the pre-trained weight files in this folder)
$ unzip -d . pretrained.zip
$ (Now, you can get a folder named pretrained)
```
Then you need to build a docker container to run the trianing file.

```
$ (You should be in the MMFSeg folder)
$ docker run -it --shm-size 8G -p 1234:6006 --name docker_container_MMFSeg --gpus all -v ~/MMFSeg:/workspace docker_image_MMFSeg
$ (currently, you should be in the docker)
$ cd /workspace
$ python3 train_C2T_NA_Conv0_Trans0.py
```

The train_C2T_NA_Conv0_Trans0.py file defaults to training MMFSeg using the Pothole-600 dataset. If you want to train MMFSeg using the NO-4K dataset, you need to change the `data_dir` parameter in the `train_C2T_NA_Conv0_Trans0.py` file to `NO4K` and adjust the `C.image_height` and `C.image_width` parameters in the `config.py` file, which represent the image resolution.



* To see the training process
```
$ (fire up another terminal)
$ docker exec -it docker_container_MMFSeg /bin/bash
$ cd /workspace
$ tensorboard --bind_all --logdir=./runs/tensorboard_log/
$ (fire up your favorite browser with http://localhost:1234, you will see the tensorboard)
```
The results will be saved in the `./runs` folder.
Note: Please change the smoothing factor in the Tensorboard webpage to `0.999`, otherwise, you may not find the patterns from the noisy plots. If you have the error `docker: Error response from daemon: could not select device driver`, please first install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) on your computer!

# Citation
If you use MMFSeg in your academic work, please cite:
```
@ARTICLE{11176062,
  author={Feng, Zhen and Guo, Yanning and Fan, Rui and Sun, Yuxiang},
  journal={IEEE Transactions on Automation Science and Engineering}, 
  title={MMFSeg: Multi-structure Multi-feature Fusion for Segmentation of Road Potholes}, 
  year={2025},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Transformers;Roads;Convolutional neural networks;Data mining;Biomedical imaging;Semantic segmentation;Fuses;Autonomous vehicles;Accuracy;Pothole Segmentation;Multi-modal Fusion;Autonomous Vehicles;Convolution-Transformer Structure},
  doi={10.1109/TASE.2025.3613629}}
```

# Acknowledgement
Some of the codes are borrowed from [IGFNet](https://github.com/lab-sun/IGFNet).
