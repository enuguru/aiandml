# broadcasting error
from numpy import array
# define two-dimensional array
A = array([
	[1, 2, 3],
	[1, 2, 3]])
print(A.shape)
# define one-dimensional array
b = array([1, 2])
print(b.shape)
# attempt broadcast
C = A + b
print(C)
