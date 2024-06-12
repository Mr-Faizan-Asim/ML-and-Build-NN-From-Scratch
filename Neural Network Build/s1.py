input = [3,6,8]
weight = [1.1,3.3,4.3]
bias = 3

output = 0
for i in range(len(input)):
    output += input[i] * weight[i]
output += bias
print(output)