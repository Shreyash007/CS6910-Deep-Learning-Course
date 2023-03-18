
# CS6910: Assignment-1
## Problem Statement
The goal of this assignment  is to implement our own feedforward, backpropagation code, use gradient descent (and its variants) with backpropagation, use our own optimizers and use it for a classification task and keep track of our
experiments using [wandb.ai](https://wandb.ai/home).


## Prerequisites

```
python 3.9
numpy 1.21.5
keras #ONLY FOR IMPORTING DATASET
```
 - Clone/download  this repository
 - I have conducted all my experiments in Google Collab, for running in google colab, install wandb using following command -
  ``` 
  !pip install wandb 
  ```
 - For running locally, install wandb and other required libraries using following command  
  ``` 
  pip install wandb
  pip install numpy
  pip install keras
  ```

## Dataset
- I have used Fashion-MNIST dataset for complete experiments.
- I have used MNIST dataset for Q10.

## Hyperparameters used in experiments
|Sr. no| Hyperparameter| Variation/values used|
|------|---------------|-----------------|
|1.| Activation function| Sigmoid, tanh,ReLu|
|2.| Loss function | Cross entropy, Mean squared error|
|3.| Initialisation| Random, Xavier|
|4.| Optimizer| Stochastic gradient descent, Momentum gradient descent, Nesterov gradient descent, RMSprop, ADAM, NADAM|
|5.| Batch size| 32, 64 ,128|
|6.| Hidden layers| [64,64,64],[128,128,128],[256,256,256],[64,64,64,64],[64,64,64,64,64],[128,128,128,128],[128,128,128,128,128]|
|7.| Epochs| 10,20,30|
|8.| Learning rate| 0.001,0.0001 |
|9.| Weight decay| 0, 0.0005 |
|10.| Dropout rate| 0, 0.1 |




## Question 1, 2, 3

The code for question 1,2,3 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/349f0e600abf3c370c902df77675dbb2577d06aa).


## Evaluation file(train.py)

For evaluating our model download [train.py](https://github.com/Shreyash007/CS6910-Deep-Learning-Course/blob/main/train.py) file. (make sure you have all the prerequisite libraries installed). To check the wandb log for evaluation run the following command in the command line(this will take the default arguments).
```
python train.py 
```
The arguments supported by train.py file are:
Supported arguments can also be found by:
```
python train.py -h
```

| Name | Default Value | Description |
| :---: | :-------------: | :----------- |
| `--wandb_project` | "CS-6910 A1" | Project name used to track experiments in Weights & Biases dashboard |
| `--wandb_entity` | "shreyashgadgil007"  | Wandb Entity used to track experiments in the Weights & Biases dashboard. |
| `--dataset` | "fashion_mnist" | choices:  ["mnist", "fashion_mnist"] |
| `--epochs` | 30 |  Number of epochs to train neural network.|
| `--batch_size` | 32 | Batch size used to train neural network. | 
| `--loss_function` | "cross_entropy" | choices:  ["square_error", "cross_entropy"] |
| `--optimiser` | "nadam" | choices:  ["gd", "mgd", "ngd", "rmsprop", "adam", "nadam"] | 
| `--learning_rate` | 0.0001 | Learning rate used to optimize model parameters | 
| `--weight_decay` | 0.0005 | Weight decay used by optimizers. |
| `--initialisation` | "xavier" | choices:  ["random", "xavier"] | 
| `--hidden_layer` | [256,256,256] | Number of hidden layers used in feedforward neural network. | 
| `--activation` | sigmoid | choices:  ["sigmoid", "tanh", "relu"] |
| `--dropout_rate` | 0.1 | choice in range (0,1) |

#### The default run has 30 epochs and  hidden layer size [256,256,256]. Hence, it may take some time to create the logs. Check the command line for the runtime.

## Report

The wandb report for this assignment can be found [here](https://wandb.ai/shreyashgadgil007/CS-6910%20A1/reports/CS6910-Assignment-1--VmlldzozNTQ1MjU1).
## Author
[Shreyash Gadgil](https://github.com/Shreyash007)
ED22S016
