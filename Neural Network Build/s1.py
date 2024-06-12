input = [3, 6, 8]
weights = [
    [1.1, 3.01, 2.3],
    [4.1, -2.1, 4.2],
    [1.6, 6.4, -1.9]
]
biases = [3, 0.5, 2]

output = []

for weight, bias in zip(weights, biases):
    o = sum(i * w for i, w in zip(input, weight)) + bias
    output.append(o)

print(output)