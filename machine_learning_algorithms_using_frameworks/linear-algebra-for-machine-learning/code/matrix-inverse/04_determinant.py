# matrix determinant
from numpy import array
from numpy.linalg import det
# define matrix
A = array([
	[1, 2, 3],
	[4, 5, 6],
	[7, 8, 9]])
print(A)
# calculate determinant
B = det(A)
print(B)