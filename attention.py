import numpy as np
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

Q = np.array([[1, 0],
              [0, 1]])

K = np.array([[1, 0],
              [0, 1]])

print("Q:")
print(Q)

print("K:")
print(K)
V = np.array([[1, 2],
              [3, 4]])

print("K transposta:")
print(K.T)

scores = Q @ K.T
print("Q x K^T:")
print(scores)
dk = Q.shape[1]

scaled_scores = scores / np.sqrt(dk)

print("Scaled Scores:")
print(scaled_scores)
attention_weights = softmax(scaled_scores)

print("Attention Weights:")
print(attention_weights)
output = attention_weights @ V

print("Output Final:")
print(output)