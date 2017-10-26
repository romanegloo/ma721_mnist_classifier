# Yet Another MNIST Classifier
* MA 721 course project -- optimizing hyper-parameters of a neural network)
* Author: Jiho Noh (jiho.noh@uky.edu)

## Description
This project is for hyper-parameter optimization with a number of modified
MNIST datasets. The goal is 1) to learn how to implement a neural network by
using a conventional deep learning platforms, and 2) to obtain practical
skills of tuning a neural network with configurable parameters

### Datasets
* **Dataset 1**  
  * [training] The first 5,000 of the original training dataset  
  * [testing] The same of training dataset

* **Dataset 2**  
  * [traning] The full entire original training dataset, but 745 is added to
    all the values. (The resulting data has entries ranging from 745 to 1000)  
  * [testing] The full entire original testing dataset

* **Dataset 3**  
  * [training] Take three subsets (0\~40,000 / 10,000\~50,000 / 20,000\~60,
  000) of
    the images and concatenate them to make three digit numbers. Labels are in
    between 0 and 999.  
  * [testing] The same as above with the subsets (0\~8,000 / 1,000\~9,000 /
    2,000\~10,000)
* **Dataset 4**  
  * [training] Add one image upon another image with two subsets from
    (0~50,000 / 10,000 ~ 60,000), the labels are in between 0 and 44 (unique
    labels for all possible combinations)  
  * [testing] The same as above from two subsets (0\~9,000 / 1,000\~10,000)

### Configurable Parameters

* **Network Capacity**  
    * number of hidden layers (--num-hidden-layers)
    * number of hidden units per layer (--num-hidden-units)
    * minibatch size (--batch-size)

* **Optimizer**
    * optimizer: sgd, ada, adamax (--optimizer)
    * learning rate (--learning-rate)

* **Initialization**
    * initialization schemes: none, uniform, xavier_normal (--weight-init)

* **Regularization**
    * default of L2 by weight decay parameter (--weight-decay)

### Methods

1. We are not using a convolution layer. We build a network with a sequence of fully connected linear layers.
2. By a random search, we seek for a network with lower capacity and higher accuracy, which is represented by the accuracy ratio to the number of parameters times the number of epochs. `ratio = [test_accuracy] / ([# of parameters] * [# of epochs])^.1`
3. Optimizer is not randomly selected. It needs to be specified. Otherwise SGD will be used.
4. Best accuracy is obtained by running the model up to 50 epochs.

### Results

| dataset | capacity (epoch x layers x units) | batch\_size | optimizer | regularization | accuracy (tr/vl/ts) | ratio (acc. to computation) | best accuracy |
|:--------:|:---------------------------------:|:----------:|:--------:|:--------------:|:--------:|:--------------------:|:-------------:|
| dataset1 | 6 x 2 x 48 | 64 | SGD (lr=2e-4) | L2 (decay=0.1) | 89.69 / 88.00 / 91.60 | 26.39 | 99.20 (48 epochs) |
| dataset2 | 2 x 4 x 128 | 512 | Adam (lr=1e-4)| L2 (decay=0.1) | 86.52 / 88.93 / 89.80 | 25.39 | 95.76 (50 epochs) |
| dataset3 | 10 x 2 x 256 | 512 | SGD (lr=3.5e-3) | L2 (decay=0.2) | 97.23 / 76.45 / 77.80 | 15.92 | 84.99 (48 epochs) |
| dataset4 | 5 x 4 x 64 | 64 | SGD (lr=5.9e-5) | L2 (decay=0.1) | 72.88 / 70.34 / 71.39 | 20.30 | 80.31 (47 epochs) |

#### Losses

<img src="https://github.com/romanegloo/ma721_mnist_classifier/blob/master/dataset1.png?raw=true" alt="Losses of Dataset 1" style="width:350px;"/>
<img src="https://github.com/romanegloo/ma721_mnist_classifier/blob/master/dataset2.png?raw=true" alt="Losses of Dataset 2" style="width:350px;"/>
<img src="https://github.com/romanegloo/ma721_mnist_classifier/blob/master/dataset3.png?raw=true" alt="Losses of Dataset 3" style="width:350px;"/>
<img src="https://github.com/romanegloo/ma721_mnist_classifier/blob/master/dataset4.png?raw=true" alt="Losses of Dataset 4" style="width:350px;"/>

### Usage

run a model with specific settings:

```
python3 scripts/pipeline.py --dataset-name dataset3 --num-hidden-layers 2 --num-hidden-units 256 --batch-size 128 --optimizer sgd --learning-rate 1e-3 --weight-init xavier_normal --weight-decay .5
```

random search for optimal settings:

```
python3 scripts/pipeline.py --dataset-name dataset3 --num-random-models 50 --weight-init xavier_normal --weight-decay .3 --early-stop
```

all options

```
usage: MNIST Handwritten Digits Classifier [-h] [--num-epochs NUM_EPOCHS]
                                           [--batch-size BATCH_SIZE]
                                           [--data-workders DATA_WORKDERS]
                                           [--dataset-name {original,dataset1,dataset2,dataset3,dataset4}]
                                           [--num-random-models NUM_RANDOM_MODELS]
                                           [--early-stop] [--no-cuda]
                                           [--gpu GPU] [--parallel PARALLEL]
                                           [--model-dir MODEL_DIR]
                                           [--data-dir DATA_DIR]
                                           [--train-img-file TRAIN_IMG_FILE]
                                           [--train-lbl-file TRAIN_LBL_FILE]
                                           [--test-img-file TEST_IMG_FILE]
                                           [--test-lbl-file TEST_LBL_FILE]
                                           [--model-name MODEL_NAME]
                                           [--input-dim INPUT_DIM]
                                           [--output-dim OUTPUT_DIM]
                                           [--num-hidden-units NUM_HIDDEN_UNITS]
                                           [--num-hidden-layers NUM_HIDDEN_LAYERS]
                                           [--optimizer OPTIMIZER]
                                           [--learning-rate LEARNING_RATE]
                                           [--weight-decay WEIGHT_DECAY]
                                           [--weight-init {none,uniform,xavier_normal}]
                                           [--print-parameters] [--draw-image]
                                           [--plot-losses] [--log-file]

optional arguments:
  -h, --help            show this help message and exit
  --model-dir MODEL_DIR
                        Directory for saved models (default:
                        /home/jno236/projects/ma721_project1/mnist_x/../model)
  --data-dir DATA_DIR   directory of datasets (default:
                        /home/jno236/projects/ma721_project1/mnist_x/../data)
  --train-img-file TRAIN_IMG_FILE
                        Decompressed MNIST train (images) file (default:
                        train-images-idx3-ubyte)
  --train-lbl-file TRAIN_LBL_FILE
                        Decompressed MNIST train (labels) file (default:
                        train-labels-idx1-ubyte)
  --test-img-file TEST_IMG_FILE
                        Decompressed MNIST test (images) file (default:
                        t10k-images-idx3-ubyte)
  --test-lbl-file TEST_LBL_FILE
                        Decompressed MNIST test (labels) file (default:
                        t10k-labels-idx1-ubyte)

Runtime:
  --num-epochs NUM_EPOCHS
                        Number of full data iterations (default: 50)
  --batch-size BATCH_SIZE
                        Batch size for training (default: 64)
  --data-workders DATA_WORKDERS
                        Number of subprocesses for data loading (default: 4)
  --dataset-name {original,dataset1,dataset2,dataset3,dataset4}
                        Name of modified datasets (default: original)
  --num-random-models NUM_RANDOM_MODELS
                        Number of random searches over hyper parameter space
                        (default: 1)
  --early-stop          Stop training when no improvements seen in 3times
                        (default: False)
  --no-cuda             Use CPU only (default: False)
  --gpu GPU             Specify GPU device id to use (default: -1)
  --parallel PARALLEL   Use DataParallel on all available GPUs (default:
                        False)
  --draw-image          draw images after test loop for verification (default:
                        False)

Model:
  --model-name MODEL_NAME
                        Unique model identifier (default: None)
  --input-dim INPUT_DIM
                        input data dimension (default: 784)
  --output-dim OUTPUT_DIM
                        output dimension (num. of classes) (default: 10)
  --num-hidden-units NUM_HIDDEN_UNITS
                        Dimension of hidden layer (default: 64)
  --num-hidden-layers NUM_HIDDEN_LAYERS
                        Number of hidden layers (default: 1)
  --optimizer OPTIMIZER
                        Optimizer: [sgd, adamax, adam] (default: sgd)
  --learning-rate LEARNING_RATE
                        Learning rate for optimizer (default: 0.0001)
  --weight-decay WEIGHT_DECAY
                        Weight decay, as L2 regularization by default
                        (default: 0)
  --weight-init {none,uniform,xavier_normal}
                        Add weight initialization scheme (default: none)

General:
  --print-parameters    Print model parameters (default: False)
  --plot-losses         plot train/test losses to epochs (default: False)
  --log-file            write logging on a file (default: False)
```
