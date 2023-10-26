# matrix-vector multiplication
from numpy import array
# define matrix
A = array([
	[1, 2],
	[3, 4],
	[5, 6]])
print(A)
# define vector
B = array([0.5, 0.5])
print(B)
# multiply
C = A.dot(B)
print(C)