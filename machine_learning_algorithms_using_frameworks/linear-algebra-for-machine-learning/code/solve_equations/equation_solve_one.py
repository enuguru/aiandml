import numpy as np
# define matrix A using Numpy arrays
A = np.array([[2, 1, 1],
 [1, 3, 2],
 [1, 0, 0]]) 

#define matrix B
B = np.array([4, 5, 6]) 

# linalg.solve is the function of NumPy to solve a system of linear scalar equations
print("Solutions:\n",np.linalg.solve(A, B ))
