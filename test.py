import numpy as np

a = np.zeros(5)
b = np.array([1, 2, 3, 4, 5])

m = np.array([[-1, -1, -1, -1, -1],
             [-2, -2, -2, -2, -2],
             [-3, -3, -3, -3, -3],
             [-4, -4, -4, -4, -4],
             [-5, -5, -5, -5, -5]])

for i in range(5):
    a[i] += np.sum(m[i])
print(a)