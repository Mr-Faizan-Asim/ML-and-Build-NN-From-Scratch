import numpy as np
from nnfs.datasets import spiral_data

X = [[3, 6, 8],
         [2,5.5,-3],
         [-1.5,2.7,3.3]]

X, y = spiral_data(samples=100,classes=3) 

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,input):
        self.output = np.dot(input,self.weights) + self.biases

class Activation_Relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Activation_Softmax:
    def forward(self,inputs):
        exp_val = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        sum_each = np.sum(exp_val,axis=1,keepdims=True)
        probabilities = exp_val / sum_each
        self.output = probabilities

       
layer1 = Layer_Dense(2,3)
layer2 = Layer_Dense(3,3)
Activation1 = Activation_Relu()
Activation2 = Activation_Softmax()

layer1.forward(X)
Activation1.forward(layer1.output)
layer2.forward(Activation1.output)
Activation2.forward(layer2.output)

print(Activation2.output)



