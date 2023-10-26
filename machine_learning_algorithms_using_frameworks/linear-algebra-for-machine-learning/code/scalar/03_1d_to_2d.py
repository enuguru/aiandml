# broadcast one-dimensional array to two-dimensional array
from numpy import array
# define two-dimensional array
A = array([
	[1, 2, 3],
	[1, 2, 3]])
print(A)
# define one-dimensional array
b = array([1, 2, 3])
print(b)
# broadcast
C = A + b
print(C)