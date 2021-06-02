# Cows2021
This repository contains the source code that accompanies our paper "Towards Self-Supervision for Video Identification of Individual Holstein-Friesian Cattle: The Cows2021 Dataset" at: https://arxiv.org/abs/2105.01938. At its core, the code in this repository is adapted and extended (with permission) from William Andrew et al's work on "Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning" published in Computers and Electronics in Agriculture 2021 -- [paper](https://arxiv.org/pdf/2006.09205.pdf), [source code](https://github.com/CWOA/MetricLearningIdentification) 

The trained weights from the for cow detection and identification are included in this repository at `weights/`.


## Installation
1) Clone this repository.
2) Install any missing requirements via pip or conda: [numpy](https://pypi.org/project/numpy/), [PyTorch](https://pytorch.org/), [OpenCV](https://pypi.org/project/opencv-python/), [Pillow](https://pypi.org/project/Pillow/), [tqdm](https://pypi.org/project/tqdm/), [sklearn](https://pypi.org/project/scikit-learn/), [seaborn](https://pypi.org/project/seaborn/). This repository requires python 3.6+
3) Instead of installing `pycocotools`, use the pycocotools in this repository.


## Cow Detection
### Testing

## Video Processing
Run the code in `make_data` one by one to obtain the training images for individual identification. The trained model weight of detection can be found `Sub-levels/3Weights/trained_model/resnet50_trained_144.h5` from [here](https://data.bris.ac.uk/data/dataset). Alternatively, you can [download](https://data.bris.ac.uk/data/dataset) this data from the folders `Sub-levels/2Identification`. 

## Individual Identification
### Testing

## Training

## Citation
