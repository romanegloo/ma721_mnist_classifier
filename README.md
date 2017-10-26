# Yet Another MNIST Classifier
* MA 721 course project -- optimizing hyper-parameters of a neural network)
* Author: Jiho Noh (jiho.noh@uky.edu)

## 1 Description
This project is for hyper-parameter optimization with a number of modified 
MNIST datasets. The goal is 1) to learn how to implement a neural network by 
using a conventional deep learning platforms, and 2) to obtain practical 
skills of tuning a neural network with configurable parameters

### 1.1 Datasets
**Dataset 1**  
* [training] The first 5,000 of the original training dataset  
* [testing] The same of training dataset

**Dataset 2**  
* [traning] The full entire original training dataset, but 745 is added to all 
the values. (The resulting data has entries ranging from 745 to 1000)  
* [testing] The full entire original testing dataset

**Dataset 3**  
* [training] Take three subsets (0~40,000 / 10,000~50,000 / 20,000~60,000) of 
the images and concatenate them to make three digit numbers. Labels are in 
between 0 and 999.  
* [testing] The same as above with the subsets (0~8,000 / 1,000~9,000 / 
2,000~10,000)

**Dataset 4**  
* [training] Add one image upon another image with two subsets from (0~50,000 / 
10,000 ~ 60,000), the labels are in between 0 and 44 (unique labels for all 
possible combinations)  
* [testing] The same as above from two subsets (0~9,000 / 1,000~10,000)

### 1.2 Configurable Parameters

**Network Capacity**  
* number of hidden layers (--num-hidden-layers)
* number of hidden units per layer (--num-hidden-units)
* minibatch size (--batch-size)

**Optimizer**
* optimizer: sgd, ada, adamax (--optimizer)
* learning rate (--learning-rate)

**Initialization**
* initialization schemes: none, uniform, xavier_normal (--weight-init)

**Regularization**
* default of L2 by weight decay parameter (--weight-decay)

### Results

| dataset | capacity (epoch x layers x units) | batch\_size | optimizer | regularization | accuracy (tr/vl/ts) | accuracy to capacity | best accuracy | 
|:--------:|:---------------------------------:|:----------:|:--------:|:--------------:|:--------:|:--------------------:|:-------------:|
| dataset1 | 6 x 4 x 32 | 64 | SGD (lr=1e-4) | None | 92.24 / 89.40 / 91.26 | 29.53 | 99.20 (48 epochs) |
| dataset2 | 7 x 4 x 128 | 512 | Adam (lr=1e-4)| L2 (decay=0.1) | 90.6 | 24.52 | 91.73 (194 epochs) |
| dataset3 | 14 x 8 x 1024 | 64 | SGD (lr=1e-4) | None | 71.35 / 72.99 | 12 .12 | ? |
| dataset4 | 5 x 4 x 64 | 64 | SGD (lr=5.9e-5) | L2 (decay=0.1) | ? / 70.34 / 71 .39 | 20.30 | 80.31 (47 epochs) |
