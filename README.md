
# CS6910- Deep Learning course
## Problem Statement
The goal of this sssignment  is to implement our own feedforward, backpropagation code, use gradient descent (and its variants) with backpropagation, use our own optimizers and use it for a classification task and keep track of our
experiments using [wandb.ai](https://wandb.ai/home).


## Prerequisites

```
Python(find out version)
Numpy(find oout version)
keras #ONLY FOR IMPORTING DATASET
```
## Dataset
- I have used Fashion-MNIST dataset for complete experiments.
- I have used MNIST dataset for Q10.

## Installation

 - Clone/download  this repository
 - I have conducted all my experiments in Google Collab, for running in google colab, install wandb using following command -
  ``` !pip install wandb ```
 - For running locally, install wandb and other required libraries using following command  
  ``` 
  pip install wandb
  pip install numpy
  pip install keras
  ```
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

## Question 4,5,6
Solution Approach:
- Get the output of feedforward backpropagation from previous question.
- Initialize one hot function to encode the labels of images.
- Implement backpropagation function.
- Initialize predictions, accurracy, loss, functions.
- Initialize gradient discent functions.
- Implement training function to use above functions.

The code for question 3 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/81c7790be2c779fb9376a0158f4adb45645c70ec).

## Question 4

Solution Approach:

 - Split the train data in the ratio of 9:1. 90% of the data is for training purpose and 10% of the data is for validation.
 - Set the sweep function of wandb by setting up different parameters in sweep_config.
 - we can see the ouput within our wandb project using the code below-
```
wandb.agent(sweep_id,train)
```

The code for question 4 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/3540f3753067f1dda62448578739f25d638d33c7).
The wandb visualisation for question 4 can be found [here](https://wandb.ai/shreekanti/confusion_matrix1/reports/Question-4--Vmlldzo1MjY2ODc).


## Question 5

The wandb visualisation for question 5 can be found [here](https://wandb.ai/rituparna_adha/assignement1/reports/Shared-panel-21-03-13-11-03-82--Vmlldzo1MjY2NzA).



## Question 6

The wandb visualisation for question 6 can be found [here](https://wandb.ai/rituparna_adha/assignement1/reports/Shared-panel-21-03-13-11-03-73--Vmlldzo1MjY2NzU).

## Question 7
Solution Approach:
- Get the best model.
- Report the best accuracy.
- The best model configuration is-
        learning_rate: 0.001,
	epochs: 10,
	no_hidden_layer: 3,
	size_hidden_layers:128,
	optimizer: adam,
	batch_size:128,
	activation: tanh,
	weight_initializations: random,
	weight_decay: 0,
	loss_function:ce

- Implement a function to calculate confusion matrix.
- Plot and integrate wandb to keep track using wandb.
The best model can be found [here](https://github.com/RituparnaAdha/cs6910/tree/main/Assignment1/model).
The code for question 7 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/46a7deb1820b546099d0d6fb43afa8eacb6cdb34).
The wandb visualisation for question 7 can be found [here](https://wandb.ai/shreekanti/confusion_matrix1?workspace=user-shreekanti).
## Question 8
Solution Approach:
- Implement a function `squared error loss`.
- Get outputs of both `squared error loss` and `cross entropy loss`.
- Integrate the outputs of `squared error loss` and `cross entropy loss` to see automatically generated plot on wandb.

The code for question 8 can be found [here](https://github.com/RituparnaAdha/cs6910/commit/dada70a3e3ff58c3eb49839d602be272318946e5).
The wandb visualisation for question 8 can be found [here](https://wandb.ai/shreekanti/assignement1-lossfunc1?workspace=user-shreekanti).
## Evaluation

The code for evaluating our test data can be found [here](https://github.com/RituparnaAdha/cs6910/blob/main/Assignment1/evaluation.py).

## Report

The report for this assignment can be found [here](https://wandb.ai/shreekanti/confusion_matrix1/reports/Assignment1--Vmlldzo1MjY1MjU).
## Authors

 - [Shree Kanti](https://github.com/shreekanti/) 
 - [Rituparna Adha](https://github.com/RituparnaAdha/)
