# MicroTensorGrad

A Tensor Based autograd engine used to train neural network based models.

## Features

- **Tensor-Based Autograd:** Automatic differentiation engine operating directly on Tensors.
- **Neural Network Training:** Built specifically to construct, backpropagate through, and train neural network architectures.

## Test
Look at test.ipynb (it shows how to use the library by training a MNIST model)

## Notice
Data (X) and Target (Y) should always be 3d array, for example if you're training an MLP model, then the data shape should be (n, m, 1), where n is the batch, m is rows, column is 1

## Installation

```bash
pip install MicroTensorGrad
```