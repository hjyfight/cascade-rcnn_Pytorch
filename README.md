An implementation of [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726). I only trained and tested on pascal voc dataset. The source code is [here](https://github.com/zhaoweicai/cascade-rcnn) which implemented by caffe and also evalated on pascal voc.

## Introduction

As we all know,  the cascade structure is designed for R-CNN structure, so i just used the cascade structure based on [DetNet](https://arxiv.org/abs/1804.06215) to train and test on pascal voc dataset (DetNet is not only faster than fpn-resnet101, but also better than fpn-resnet101).

Based on [**DetNet_Pytorch**](https://github.com/guoruoqian/DetNet_pytorch), i mainly changed the forward function in fpn.py. It‘s just a naive implementation, so its speed is not fast. 

## Update

**2019/01/01:** 

- [x] Fix bugs in demo, now you can run demo.py file. **Note the default demo.py merely support pascal_voc categories.**  You need to change the ```pascal_classes``` in demo.py to adapt your own dataset. If you want to know more details, please see the **usage** part.
- [x] upload the pretrained DetNet59-Cascade which in below table. 

## Benchmarking

I benchmark this code thoroughly on pascal voc2007 and 07+12. Below are the results:

1). PASCAL VOC 2007 (Train/Test: 07trainval/07test, scale=600, ROI Align)

| model（FPN）     | GPUs            | Batch Size | lr   | lr_decay | max_epoch | Speed/epoch | Memory/GPU | AP   | AP50 | AP75 |
| ---------------- | --------------- | ---------- | ---- | -------- | --------- | ----------- | ---------- | ---- | ---- | ---- |
| DetNet59         | 1 GTX 1080 (Ti) | 2          | 1e-3 | 10       | 12        | 0.89hr      | 6137MB     | 44.8 | 76.1 | 46.2 |
| DetNet59-Cascade | 1 GTX 1080 (Ti) | 2          | 1e-3 | 10       | 12        | 1.62hr      | 6629MB     | 48.9 | 75.9 | 53.0 |

2). PASCAL VOC 07+12 (Train/Test: 07+12trainval/07test, scale=600, ROI Align)

| model（FPN）                                                 | GPUs            | Batch Size | lr   | lr_decay | max_epoch | Speed/epoch | Memory/GPU | AP   | AP50 | AP75 |
| ------------------------------------------------------------ | --------------- | ---------- | ---- | -------- | --------- | ----------- | ---------- | ---- | ---- | ---- |
| DetNet59                                                     | 1 GTX 1080 (Ti) | 1          | 1e-3 | 10       | 12        | 2.41hr      | 9511MB     | 53.0 | 80.7 | 58.2 |
| [DetNet59-Cascade](https://drive.google.com/open?id=1AUBe1oIwCMVai2cIPIlZx-EgEtvMSYs-) | 1 GTX 1080 (Ti) | 1          | 1e-3 | 10       | 12        | 4.60hr      | 1073MB     | 55.6 | 80.1 | 61.0 |

## Environment Setup

This project has been validated on Ubuntu 16.04/18.04 with NVIDIA GPUs. Before installing Python packages, ensure that your system provides:

- An NVIDIA GPU with CUDA 8.0+ and cuDNN installed
- GCC 5 or newer together with the standard build toolchain

You can install the required build packages on Ubuntu with:

```shell
sudo apt-get update
sudo apt-get install build-essential python3-dev libglib2.0-0 libsm6 libxext6 libxrender-dev
```

Follow the steps below to configure the Python environment:

1. **Clone the repository and create the data directory**

   ```shell
   git clone https://github.com/guoruoqian/cascade-rcnn_Pytorch.git
   cd cascade-rcnn_Pytorch
   mkdir -p data
   ```

2. **Create and activate a Python environment**

   The code base targets Python 3.6. Create an isolated environment using either Conda or `venv`:

   ```shell
   conda create -n cascade-rcnn python=3.6
   conda activate cascade-rcnn
   ```
   _or_
   ```shell
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install PyTorch**

   Cascade R-CNN relies on PyTorch 0.3.x. Install the wheel compatible with your CUDA toolkit (CUDA 8.0 shown below):

   ```shell
   pip install https://download.pytorch.org/whl/cu80/torch-0.3.1-cp36-cp36m-linux_x86_64.whl
   pip install https://download.pytorch.org/whl/cu80/torchvision-0.2.0-py2.py3-none-any.whl
   ```

   For different CUDA versions or CPU-only builds, refer to the [PyTorch previous versions archive](https://pytorch.org/get-started/previous-versions/).

4. **Install the remaining Python dependencies**

   ```shell
   pip install -r requirements.txt
   ```

5. **Build the CUDA/C++ extensions**

   ```shell
   cd lib
   sh make.sh
   cd ..
   ```

   Make sure the `-arch` flag in `make.sh` matches your GPU architecture (see the reference table below).

6. **(Optional) Verify the installation**

   ```shell
   python - <<'PY'
   import torch
   from model.utils.cython_nms import nms
   print("Cascade R-CNN environment ready.")
   PY
   ```

> **GPU architecture quick reference**

| GPU model                  | Architecture |
| :------------------------- | :----------: |
| TitanX (Maxwell/Pascal)    |    sm_52     |
| GTX 960M                   |    sm_50     |
| GTX 1080 (Ti)              |    sm_61     |
| Grid K520 (AWS g2.2xlarge) |    sm_30     |
| Tesla K80 (AWS p2.xlarge)  |    sm_37     |

### Data Preparation

- **VOC 2007**: Please follow the instructions in [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn#beyond-the-demo-installation-for-training-and-testing-models) to prepare the VOC dataset. After downloading the data, create soft links in the `data/` folder.
- **VOC 07 + 12**: Please follow the instructions in [YuwenXiong/py-R-FCN](https://github.com/YuwenXiong/py-R-FCN/blob/master/README.md#preparation-for-training--testing). **This guide is often more helpful for preparing VOC datasets.**

### Pretrained Model 

You can download the detnet59 model which I trained on ImageNet from:

- detnet59: [dropbox](https://www.dropbox.com/home/DetNet?preview=detnet59.pth)，[baiduyun](https://pan.baidu.com/s/14_ztsAKcrZGb4nnm8aCMyQ)，[google drive](https://drive.google.com/open?id=1kKgjhpdb5ruoGkpZuEQ-ZIZkbxMtWZIQ)

Download it and put it into `data/pretrained_model/`.

## Usage

If you want to use cascade structure, you must set  `--cascade`  and  `--cag` in the below script. `cag` determine whether perform class_agnostic bbox regression. 

train voc2007 use cascade structure:

```shell
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name --dataset pascal_voc --net detnet59 --bs 2 --nw 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True --cag --cascade
```

test voc2007:

```shell
CUDA_VISIBLE_DEVICES=3 python3 test_net.py exp_name --dataset pascal_voc --net detnet59 --checksession 1 --checkepoch 7 --checkpoint 5010 --cuda --load_dir weights --cag --cascade
```

Before training voc07+12, you must set ASPECT_CROPPING in detnet59.yml False, or you will encounter some error during the training. 

train voc07+12:

```shell
CUDA_VISIBLE_DEVICES=3 python3 trainval_net.py exp_name2 --dataset pascal_voc_0712 --net detnet59 --bs 1 --nw 4 --lr 1e-3 --epochs 12 --save_dir weights --cuda --use_tfboard True --cag --cascade
```
run demo.py :

Before run demo, you must make dictionary 'demo_images' and put images (VOC images) in it. You can download the pretrained model  listed in above tables.  

```shell
CUDA_VISIBLE_DEVICES=3 python3 demo.py exp_name2 --dataset pascal_voc_0712 --net detnet59 --checksession 1 --checkepoch 8 --checkpoint 33101 --cuda --load_dir weights --cag --image_dir demo_images --cascade --result_dir vis_cascade
```

