import math
import numpy as np
#Basic
Layer = [4.5, -1.5, 2.3]

output = [math.exp(i) for i in Layer]

outputSum = sum(output)

normVal = [i/outputSum for i in output]
#Simple


X = [[3, 6, 8],
         [2,5.5,-3],
         [-1.5,2.7,3.3]]

exp_val = np.exp(X)
sum_each = np.sum(exp_val,axis=1,keepdims=True)
probabilities = exp_val / sum_each


