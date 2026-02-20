import numpy as np
from attention import scaled_dot_product_attention

Q = np.array([[1, 0],
              [0, 1]])

K = np.array([[1, 0],
              [0, 1]])

V = np.array([[1, 2],
              [3, 4]])

output = scaled_dot_product_attention(Q, K, V)

print("Resultado da Attention:")
print(output)