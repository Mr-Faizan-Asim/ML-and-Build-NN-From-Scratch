import math

output = [0.7,0.1,0.1]
target = [1,0,0]
loss = -sum(target[i] * math.log(output[i]) for i in range(len(target)))

print(loss)