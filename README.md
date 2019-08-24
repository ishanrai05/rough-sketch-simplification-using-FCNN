# Rough Sketch Simplification using FCNN in PyTorch

This repository contains code of the paper [Learning to Simplify: Fully Convolutional Networks for Rough Sketch Cleanup](http://www.f.waseda.jp/hfs/SimoSerraSIGGRAPH2016.pdf) which is tested and trained on cutom datasets. It is based on PyTorch.

## Overview

The paper presents novel technique to simplify sketch drawings based on learning a series of convolution operators. Image of any dimension can be fed into the network, and it outputs the image of same dimension as the input image.

![model](images/model.png)

The architecture consists of encoder and a decoder, the first part acts as an encoder and spatially compresses the image, the second part, processes and extracts the essential lines from the image, and the third and last part acts as a decoder which converts the small more simple representation to an grayscale image of the same resolution as the input. This is all done using convolutions.
The down- and up-convolution architecture may seem similar to a simple filter banks. However, it is important to realize that the number of channels is much larger where resolution is lower, e.g., 1024 where the size is 1/8. This ensures that information that leads to clean lines is carried through the low-resolution part; the network is trained to choose which information to carry by the encoder- decoder architecture. Padding is used to compensate for the kernel size and ensure the output is the same size as the input when a stride of 1 is used. Pooling layers are replaced by convolutional layers with increased strides to lower the resolution from the previous layer.



## Contents
- [Rough Sketch Simplification using FCNN in PyTorch](#Rough-Sketch-Simplification-using-FCNN-in-PyTorch)
  - [Overview](#Overview)
  - [Contents](#Contents)
  - [1. Setup Instructions and Dependencies](#1-Setup-Instructions-and-Dependencies)
  - [2. Dataset](#2-Dataset)
  - [3. Training the model](#3-Training-the-model)
  - [5. Model Architecture](#5-Model-Architecture)
  - [6. Observations](#6-Observations)
    - [Training](#Training)
    - [Predicitons](#Predicitons)
  - [7. Repository overview](#7-Repository-overview)


## 1. Setup Instructions and Dependencies

Clone the repositiory on your local machine.

``` Batchfile
git clone https://github.com/ishanrai05/rough-sketch-simplification-using-FCNN
```

Start a virtual environment using python3
``` Batchfile
virtualenv env
```


Install the dependencies
``` Batchfile
pip install -r requirements.txt
```

You can also use google collab notebook. In that case just upload the notebook provided in the repository and you are good to go.


## 2. Dataset

The authors have not provided dataset for the paper. So I created my own. I have uploaded the dataset on drive, the link to which can be found [here](https://drive.google.com/open?id=14NQTqITAiw8o-JgdnumQ-K0asLRwJy7q). Feel free to use it.

## 3. Training the model

To train the model, run

```Batchfile
python main.py --train=True
```

optional arguments:
  | argument | default | desciption|
  | --- | --- | --- |
  | -h, --help | | show help message and exit |
  | --use_cuda | False | device to train on. default is CPU |
  | --samples | False | See sample images |
  | --num_epochs | 10 | Number of epochs to train on |
  | --train | True | train the model |
  | --root | '.' | Root Directory for Input and Target images |


## 5. Model Architecture

![archi](images/archi.png)  

## 6. Observations

The modal takes about 63 mins to train for 150 epochs on Google Collab with Nvidia Tesla K80 GPU.

### Training

| Epoch | Prediction |
| --- | --- |
| 2 | ![epoch2](pred/2.png) |
| 60 | ![epoch40](pred/60.png) |
| 100 | ![epoch80](pred/100.png) |
| 140 | ![epoch120](pred/140.png) |

### Predicitons

![pred1](pred/pred1.png)
![pred2](pred/pred2.png)
![pred3](pred/pred3.png)

## 7. Repository overview


This repository contains the following files and folders

1. **notebook**: This folder contains the jupyter notebook for code.

2. **images**: Contains images.

2. **pred**: Contains prediction images.

3. `constants.py`: image width and size while training.

4. `CustomDataset.py`: code for dataset generation.

5. `model.py`: code for model as described in the paper.

5. `predict.py`: function to simplify image using model.

5. `read_data.py`: code to read images.

5. `visualize.py`: code for visualizations.

6. `utils.py`: Contains helper functions.

7. `train.py`: function to train models from scratch.

8. `main.py`: contains main code to run the model.

9.  `requirements.txt`: Lists dependencies for easy setup in virtual environments.

