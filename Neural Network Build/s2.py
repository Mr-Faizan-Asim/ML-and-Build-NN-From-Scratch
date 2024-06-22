import numpy as np
from nnfs.datasets import spiral_data

X = [[3, 6, 8],
         [2,5.5,-3],
         [-1.5,2.7,3.3]]

X, y = spiral_data(100,3) 

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,input):
        self.output = np.dot(input,self.weights) + self.biases

class Activation_Relu:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

layer1  = Layer_Dense(2,5)
activate_function = Activation_Relu()
layer1.forward(X)
print(layer1.output)
print("Result With Activation Function")
activate_function.forward(layer1.output)
print(activate_function.output)

