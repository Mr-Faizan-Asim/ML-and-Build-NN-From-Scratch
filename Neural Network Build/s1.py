import numpy as np

input = [[3, 6, 8],
         [2,5.5,-3],
         [-1.5,2.7,3.3]]
weights = [
    [1.1, 3.01, 2.3],
    [4.1, -2.1, 4.2],
    [1.6, 6.4, -1.9]
]
biases = [3, 0.5, 2]
biases_layer2 = np.array([1, -1, 2.5])

layer_1 =  np.dot(input,weights) + biases
layer_2 =  np.dot(layer_1,weights) + biases_layer2
print(layer_2)