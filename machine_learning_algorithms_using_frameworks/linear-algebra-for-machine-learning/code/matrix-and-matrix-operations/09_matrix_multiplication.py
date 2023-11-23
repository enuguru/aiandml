# input two matrices of size n x m
matrix1 = [[12,7,3],
		[4 ,5,6],
		[7 ,8,9]]
matrix2 = [[5,8,1],
		[6,7,3],
		[4,5,9]]

res = [[0 for x in range(3)] for y in range(3)] 

# explicit for loops
for i in range(len(matrix1)):
	for j in range(len(matrix2[0])):
		for k in range(len(matrix2)):

			# resulted matrix
			res[i][j] += matrix1[i][k] * matrix2[k][j]

print (res)
