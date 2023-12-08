# matrix determinant
from numpy import array
from numpy.linalg import det
# define matrix
A = array([
	[7, -4, 2],
	[3, 1, -5],
	[2, 2, -5]])
print(A)
# calculate determinant
B = det(A)
print(B)
