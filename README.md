# Cows2021
This repository contains the source code that accompanies our paper "Towards Self-Supervision for Video Identification of Individual Holstein-Friesian Cattle: The Cows2021 Dataset" at: https://arxiv.org/abs/2105.01938. At its core, the code in this repository is adapted and extended (with permission) from William Andrew et al's work on "Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning" published in Computers and Electronics in Agriculture 2021 -- [paper](https://arxiv.org/pdf/2006.09205.pdf), [source code](https://github.com/CWOA/MetricLearningIdentification) 

The trained weights from the for cow detection and identification are included in this repository at `weights/`.


## Depedencies
1) Clone this repository.
2) Modify the prefix in `conda_detection.yaml` with your own path and name. Create your environment using anaconda. 
3) Instead of installing `pycocotools`, use the pycocotools in this repository.

## Usage

### Cow Detection

#### Testing
To test a trained model, copy the [trained model weight for detection](https://data.bris.ac.uk/data/dataset/0096ed43188f439745155da595f13b38) resnet50_trained_144.h5 to `Detection/test/trained_model`. Run rotate_test.py.


#### Training


### Training Data for individual identification and Video Processing
[Download](https://data.bris.ac.uk/data/dataset/44ec2bfeda051bf39f8357d237db03af) training data from `Sub-levels/Identification/Train`. Alternatively, you can generate the trainning data from [raw videos](https://data.bris.ac.uk/data/dataset/4vnrca7qw1642qlwxjadp87h7) from `Sub-levels/Identification/Videos`. Run the code in `make_data` one by one. You can found training images in a folder called `Crop_split`. When running the codes, you will need the [trained model weight of detection](https://data.bris.ac.uk/data/dataset/0096ed43188f439745155da595f13b38), which can be found in `Sub-levels/3Weights/trained_model/resnet50_trained_144.h5`. 

### Individual Identification
#### Testing
To test a trained model by inferring embeddings and using GMM to obtain the accuracy, run the code in `Test` one by one.

#### Training
To train the model, use python train.py -h to get help with setting command line arguments. A minimal example would be python train.pyxxx

## Citation

Consider citing ours and William's works in your own research if this repository has been useful:

```
@article{gao2021towards,
  title={Towards Self-Supervision for Video Identification of Individual Holstein-Friesian Cattle: The Cows2021 Dataset},
  author={Gao, Jing and Burghardt, Tilo and Andrew, William and Dowsey, Andrew W and Campbell, Neill W},
  journal={arXiv preprint arXiv:2105.01938},
  year={2021}
}

@article{andrew2020visual,
  title={Visual Identification of Individual Holstein Friesian Cattle via Deep Metric Learning},
  author={Andrew, William and Gao, Jing and Campbell, Neill and Dowsey, Andrew W and Burghardt, Tilo},
  journal={arXiv preprint arXiv:2006.09205},
  year={2020}
}
```

![Footer](https://github.com/Wormgit/Cows2021/tree/main/images/ids.png)
