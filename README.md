# Neural Network from Scratch

This repository contains the implementation of a simple neural network from scratch using Python and NumPy. The neural network is designed to classify spiral data into three classes. This project demonstrates the basic building blocks of a neural network, including dense layers, activation functions, and a loss function, all implemented without using any deep learning frameworks like TensorFlow or PyTorch.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)

## Overview

The goal of this project is to build a neural network that can classify spiral data. The neural network consists of:

1. Dense (fully connected) layers
2. ReLU activation function
3. Softmax activation function
4. Categorical Cross-Entropy loss function

The spiral data is generated using the `nnfs` library, which ensures reproducibility of the random data points.

## Installation

To run this project, you need to have Python installed. You can install the required libraries using pip:

```bash
pip install numpy nnfs
