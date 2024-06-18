import numpy as np

X = [[3, 6, 8],
         [2,5.5,-3],
         [-1.5,2.7,3.3]]

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,input):
        self.output = np.dot(input,self.weights) + self.biases



layer1  = Layer_Dense(3,3)
layer1.forward(X)
print(layer1.output)

print("2nd")
layer2 = Layer_Dense(3,5)
layer2.forward(X)
print(layer2.output)




