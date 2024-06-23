import numpy as np
from nnfs.datasets import spiral_data

# Import the required libraries
import nnfs
nnfs.init()  # Initialize nnfs to make the randomization reproducible

# Generate spiral data
X, y = spiral_data(samples=100, classes=3)

# Define the Dense layer class
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

# Define the ReLU activation function class
class Activation_Relu:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

# Define the Softmax activation function class
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

# Define the base Loss class
class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

# Define the Categorical Cross-Entropy loss class
class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
        negative_log_likelihoods = -np.log(correct_confidences)  # Negate the log to calculate loss
        return negative_log_likelihoods

# Create and initialize the layers and activation functions
layer1 = Layer_Dense(2, 3)
layer2 = Layer_Dense(3, 3)
activation1 = Activation_Relu()
activation2 = Activation_Softmax()

# Perform a forward pass through the network
layer1.forward(X)
activation1.forward(layer1.output)
layer2.forward(activation1.output)
activation2.forward(layer2.output)

# Print the output of the last activation function (Softmax)
print(activation2.output)

# Calculate and print the loss
loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(activation2.output, y)
print("Loss:", loss)
