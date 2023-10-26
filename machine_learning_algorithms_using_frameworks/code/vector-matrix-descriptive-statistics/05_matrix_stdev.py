# matrix standard deviation
from numpy import array
from numpy import std
# define matrix
M = array([
	[1,2,3,4,5,6],
	[1,2,3,4,5,6]])
print(M)
# column standard deviations
col_std = std(M, ddof=1, axis=0)
print(col_std)
# row standard deviations
row_std = std(M, ddof=1, axis=1)
print(row_std)