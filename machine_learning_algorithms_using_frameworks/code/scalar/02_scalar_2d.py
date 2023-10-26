# broadcast scalar to two-dimensional array
from numpy import array
# define array
A = array([
	[1, 2, 3],
	[1, 2, 3]])
print(A)
# define scalar
b = 2
print(b)
# broadcast
C = A + b
print(C)