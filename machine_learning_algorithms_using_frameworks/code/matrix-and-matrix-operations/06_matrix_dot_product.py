# matrix dot product
from numpy import array
# define first matrix
A = array([
	[1, 2],
	[3, 4],
	[5, 6]])
print(A)
# define second matrix
B = array([
	[1, 2],
	[3, 4]])
print(B)
# multiply matrices
C = A.dot(B)
print(C)
# multiply matrices with @ operator
D = A @ B
print(D)