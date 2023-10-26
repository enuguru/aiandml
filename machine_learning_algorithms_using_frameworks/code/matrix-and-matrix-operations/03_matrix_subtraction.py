# matrix subtraction
from numpy import array
# define first matrix
A = array([
	[1, 2, 3],
	[4, 5, 6]])
print(A)
# define second matrix
B = array([
	[0.5, 0.5, 0.5],
	[0.5, 0.5, 0.5]])
print(B)
# subtract matrices
C = A - B
print(C)