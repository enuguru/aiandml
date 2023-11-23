# vector L1 norm
from numpy import array
from numpy.linalg import norm
# define vector
a = array([1, 2, 3])
print(a)
print(a.shape)
# calculate norm
l1 = norm(a, 1)
print(l1)
