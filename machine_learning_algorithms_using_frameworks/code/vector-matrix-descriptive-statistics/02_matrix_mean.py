# matrix means
from numpy import array
from numpy import mean
# define matrix
M = array([
	[1,2,3,4,5,6],
	[1,2,3,4,5,6]])
print(M)
# column means
col_mean = mean(M, axis=0)
print(col_mean)
# row means
row_mean = mean(M, axis=1)
print(row_mean)