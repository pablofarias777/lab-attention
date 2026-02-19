import numpy as np

Q = np.array([[1, 0],
              [0, 1]])

K = np.array([[1, 0],
              [0, 1]])

print("Q:")
print(Q)

print("K:")
print(K)
print("K transposta:")
print(K.T)

scores = Q @ K.T
print("Q x K^T:")
print(scores)
dk = Q.shape[1]

scaled_scores = scores / np.sqrt(dk)

print("Scaled Scores:")
print(scaled_scores)