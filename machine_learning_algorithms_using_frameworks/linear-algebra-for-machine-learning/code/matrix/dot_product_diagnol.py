import numpy as np
A = np.matrix('1 2 3; 4 5 6; 7,8,9')
print("Matrix is :\n", A)
#diagonal values
print("Diagonal of matrix A :\n", A.diagonal(0,0,1))
#dot product
print("Dot product of matrix A with 2 :\n", A.dot(2))