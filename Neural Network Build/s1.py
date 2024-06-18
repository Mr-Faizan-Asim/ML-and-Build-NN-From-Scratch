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

output =  np.dot(input,weights) + biases

print(output)