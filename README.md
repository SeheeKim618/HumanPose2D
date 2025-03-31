# [A Fusion Model based on CNN-Vision Transformer for HumanPose Estimation (KSC 2022)](https://drive.google.com/file/d/1sC1Li9IQlDmLiRUhiMM4VQVLzrb6uXsS/view?usp=sharing)

## Introduction
This is the official code of [A Fusion Model based on CNN-Vision Transformer for HumanPose Estimation (KSC 2022)](https://drive.google.com/file/d/1sC1Li9IQlDmLiRUhiMM4VQVLzrb6uXsS/view?usp=sharing), by Sehee Kim and Junhee Lee.  
CNN-based models have achieved remarkable performance in the human pose estimation task. However, it remains unclear how they capture spatial dependencies for keypoint localization. On the other hand, Vision Transformer-based models efficiently capture long-range relationships through attention layers, but tend to have significantly more parameters and higher model complexity compared to CNN-based models. This paper proposes a hybrid model that combines CNN and Vision Transformer architectures to improve keypoint estimation accuracy while reducing model redundancy. The proposed CNN–Vision Transformer hybrid model combines HigherHRNet and DaViT-Tiny using a Feature Fusion Block.

![Illustrating the architecture of the our's model](/figures/figure_arch.png)

## Main Results
### Results on COCO val2017 without multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------| 
| HigherHRNet        | HRNet-w32  | 512      |  28.6M  | 47.9   | 67.1  | 86.2  |  73.0  |  61.5  |  76.1  | 
| Ours               | HRNet-w32  | 512      |  35.5M  | 330.5  | **67.5**  | **86.9**  |  **73.6**  |  **61.9**  |  75.9  |  

### Results on COCO test-dev2017 without multi-scale test
| Method             | Backbone | Input size | #Params | GFLOPs |    AP | Ap .5 | AP .75 | AP (M) | AP (L) |
|--------------------|----------|------------|---------|--------|-------|-------|--------|--------|--------|
| OpenPose\*         |    -     | -          |   -     |  -     | 61.8  | 84.9  |  67.5  |  57.1  |  68.2  | 
| Hourglass          | Hourglass  | 512      | 277.8M  | 206.9  | 56.6  | 81.8  |  61.8  |  49.8  |  67.0  | 
| PersonLab          | ResNet-152  | 1401    |  68.7M  | 405.5  | 66.5  | 88.0  |  72.6  |  62.4  |  72.3  |
| PifPaf             |    -     | -          |   -     |  -     | 66.7  | -     |  -     |  62.4  |  72.9  | 
| Bottom-up HRNet    | HRNet-w32  | 512      |  28.5M  | 38.9   | 64.1  | 86.3  |  70.4  |  57.4  |  73.9  | 
| HigherHRNet    | HRNet-w32  | 512      |  28.6M  | 47.9   | 66.4  | 87.5  |  72.8  |  61.2  |  74.2  |
| Ours           | HRNet-w32  | 512      |  35.5M  | 330.5   | **66.8**  | **88.2**  |  **73.6**  |  **61.6**  |  74.2  |
## Environment
The code is developed using python 3.8 on Ubuntu. NVIDIA GPUs are needed. The code is developed and tested using 4 NVIDIA 3090 GPU cards. Other platforms or GPU cards are not fully tested.

## Quick start
### Installation
1. Install pytorch >= v1.1.0 following [official instruction](https://pytorch.org/).  
   - **Tested with pytorch v1.4.0**
2. Clone this repo, and we'll call the directory that you cloned as ${POSE_ROOT}.
3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Install [COCOAPI](https://github.com/cocodataset/cocoapi):
   ```
   # COCOAPI=/path/to/clone/cocoapi
   git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
   cd $COCOAPI/PythonAPI
   # Install into global site-packages
   make install
   # Alternatively, if you do not have permissions or prefer
   # not to install the COCO API into global site-packages
   python3 setup.py install --user
   ```
   Note that instructions like # COCOAPI=/path/to/install/cocoapi indicate that you should pick a path where you'd like to have the software cloned and then set an environment variable (COCOAPI in this case) accordingly.
5. Install [CrowdPoseAPI](https://github.com/Jeff-sjtu/CrowdPose) exactly the same as COCOAPI.  
   - **There is a bug in the CrowdPoseAPI, please reverse https://github.com/Jeff-sjtu/CrowdPose/commit/785e70d269a554b2ba29daf137354103221f479e**
6. Init output(training model output directory) and log(tensorboard log directory) directory:

   ```
   mkdir output 
   mkdir log
   ```

   Your directory tree should look like this:

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── tools 
   ├── README.md
   └── requirements.txt
   ```

7. Download pretrained models from our model ([GoogleDrive](https://drive.google.com/drive/folders/1MTZMFmaX1vPM8WShD6lPzrdghvcl1FAi?usp=sharing))
   ```
   ${POSE_ROOT}
    `-- models
        `-- pytorch
            `-- pose_coco
                `-- model_best.pth.tar

   ```
   
### Data preparation

**For COCO data**, please download from [COCO download](http://cocodataset.org/#download), 2017 Train/Val is needed for COCO keypoints training and validation.
Download and extract them under {POSE_ROOT}/data, and make them look like this:
```
${POSE_ROOT}
|-- data
`-- |-- coco
    `-- |-- annotations
        |   |-- person_keypoints_train2017.json
        |   `-- person_keypoints_val2017.json
        `-- images
            |-- train2017
            |   |-- 000000000009.jpg
            |   |-- 000000000025.jpg
            |   |-- 000000000030.jpg
            |   |-- ... 
            `-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
                |-- 000000000632.jpg
                |-- ... 
```

### Training and Testing

#### Testing on COCO val2017 dataset using pretrained models ([GoogleDrive](https://drive.google.com/drive/folders/1MTZMFmaX1vPM8WShD6lPzrdghvcl1FAi?usp=sharing))
 

For single-scale testing:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/model_best.pth.tar
```

By default, we use horizontal flip. To test without flip:

```
python tools/valid.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    TEST.MODEL_FILE models/pytorch/pose_coco/model_best.pth.tar \
    TEST.FLIP_TEST False
```


#### Training on COCO train2017 dataset

```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```

By default, it will use all available GPUs on the machine for training. To specify GPUs, use

```
CUDA_VISIBLE_DEVICES=0,1 python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml 
```

#### Mixed-precision training
Due to large input size for bottom-up methods, we use mixed-precision training to train our Higher-HRNet by using the following command:
```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    FP16.ENABLED True FP16.DYNAMIC_LOSS_SCALE True
```

#### Synchronized BatchNorm training
If you have limited GPU memory, please try to reduce batch size and use SyncBN to train our Higher-HRNet by using the following command:
```
python tools/dist_train.py \
    --cfg experiments/coco/higher_hrnet/w32_512_adam_lr1e-3.yaml \
    FP16.ENABLED True FP16.DYNAMIC_LOSS_SCALE True \
    MODEL.SYNC_BN True
```

## Citation
If you find this work or code is helpful in your research, please cite:
````
@inproceedings{cheng2020bottom,
  title={HigherHRNet: Scale-Aware Representation Learning for Bottom-Up Human Pose Estimation},
  author={Bowen Cheng and Bin Xiao and Jingdong Wang and Honghui Shi and Thomas S. Huang and Lei Zhang},
  booktitle={CVPR},
  year={2020}
}

@inproceedings{SunXLW19,
  title={Deep High-Resolution Representation Learning for Human Pose Estimation},
  author={Ke Sun and Bin Xiao and Dong Liu and Jingdong Wang},
  booktitle={CVPR},
  year={2019}
}

@article{wang2019deep,
  title={Deep High-Resolution Representation Learning for Visual Recognition},
  author={Wang, Jingdong and Sun, Ke and Cheng, Tianheng and Jiang, Borui and Deng, Chaorui and Zhao, Yang and Liu, Dong and Mu, Yadong and Tan, Mingkui and Wang, Xinggang and Liu, Wenyu and Xiao, Bin},
  journal={TPAMI},
  year={2019}
}
````

