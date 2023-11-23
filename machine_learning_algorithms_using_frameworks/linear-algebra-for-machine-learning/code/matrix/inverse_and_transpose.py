import numpy as np
A = np.matrix('1 2 3; 4 5 6')
print("Matrix is :\n", A)
#Transpose of matrix
print("The transpose of matrix A is :\n", A.getT())
#Complex conjugate transpose
print("Complex transpose of matrix A is :\n", A.getH())
#Multiplicative inverse
print("Multiplicative inverse of matrix A is :\n", A.getI())