import numpy as np
from fractions import Fraction as F
from math import ceil, floor, gcd, sqrt

def read_matrix(file,d):
	f = open(file,"r")
	matrix = np.full([d,d],F(0,1))
	lines = f.readlines()
	for i in range(len(lines)):
		c = []
		line = lines[i]
		j = 0
		while j <= len(line)-2:
			x = line[j]
			if x == ' ':
				j += 1
			elif x == '-':
				k = j + 1
				while line[k] != ' ' and line[k] != '/':
					k += 1
				if line[k] == ' ':
					c.append(F(int(line[j:k]),1))
					j = k + 1
				elif line[k] == '/':
					l = k + 1
					while line[l] != ' ':
						l += 1
					c.append(F(int(line[j:k]),int(line[k+1:l])))
					j = l + 1
			else:
				k = j + 1
				while line[k] != ' ' and line[k] != '/':
					k += 1
				if line[k] == ' ':
					c.append(F(int(line[j:k]),1))
					j = k + 1
				elif line[k] == '/':
					l = k + 1
					while line[l] != ' ':
						l += 1
					c.append(F(int(line[j:k]),int(line[k+1:l])))
					j = l + 1
		matrix[i,:] = c
	f.close() 
	return matrix

def write_matrix(matrix,file,mode = 'write'):
	if mode not in ['write', 'append']:
		print(mode,' is not a valid writing mode, please choose either "write" (by default) or "append"')
	row = len(matrix)
	col = len(matrix[0])
	if mode == 'write':
		f = open(file,"w")
	elif mode == 'append':
		f = open(file,"a")
	for i in range(row):
		for j in range(col):
			f.write(str(matrix[i][j]))
			f.write(' ')
		if i < row - 1:
			f.write('\n')
	f.close()

def fractional_part(number):
	num = number.numerator
	den = number.denominator
	r = num % den
	return F(r,den)
	
#Converts an integral matrix to a matrix with fractional entries with denominator 1. 
#Mostly used, so the functions here work smoothly
def convert_matrix_fractional(matrix):
	col = len(matrix)
	row = len(matrix[0])
	matrix_fraction = np.full([col,row],F(0,1))
	for i in range(col):
		for j in range(row):
			matrix_fraction[i][j] = F(int(matrix[i][j]),1)
	return matrix_fraction

#Converts an integral matrix with fractional entries to a matrix with integer entries. 
#Mostly used for prettier output or for further work.
def convert_matrix_integral(matrix):
	if type(matrix[0][0]) != F:
		print(matrix, ' does have some non fractional entries')
		return 0
	col = len(matrix)
	row = len(matrix[0])
	matrix_integral = np.zeros([col,row])
	for i in range(col):
		for j in range(row):
			if matrix[col,row].denominator != 1:
				print(matrix, ' is not integral')
				return 0
			matrix_integral[i][j] = matrix[i][j].numerator
	return matrix_integral

#Computes the smallest multiple of a matrix, such that all entries are integral.
#Specifically such that the gcd of all nonzero entries is 1.
def scale_to_integer_matrix(matrix):
	col = len(matrix)
	row = len(matrix[0])
	denom = list(dict.fromkeys([matrix[i][j].denominator for i in range(row) for j in range(col)]))
	if len(denom) == 1:
		g = 1
	else:
		g = denom[0]
		for i in range(1,len(denom)):
			g = gcd(g,denom[i])
	prod = denom[0]
	for i in range(1,len(denom)):
		prod *= denom[i]
	lcm = F(abs(prod),g)
	matrix = np.array([(lcm * matrix[i][j]).numerator for i in range(row) for j in range(col)]).reshape(row,col)
	num = list(dict.fromkeys(matrix.reshape(col * row)))
	if len(num) == 1:
		g = 1
	else:
		g = num[0]
		for i in range(1,len(num)):
			g = gcd(g,num[i])
	return F(1,g) * matrix

#Same as the matrix version.
def convert_vector_fractional(vector):
	d = len(vector)
	vector_fraction = np.full(d,F(0,1))
	for i in range(d):
		vector_fraction[i] = F(vector[i],1)
	return vector_fraction

#Same as the matrix version.	
def convert_vector_integral(vector):
	d = len(vector)
	vector_integral = np.zeros(d)
	for i in range(d):
		if vector[i].denominator != 1:
			print(vector, ' is not integral')
			return 0
		vector_integral[i] = vector[i].numerator
	return vector_integral

#Same as the matrix version.
def scale_to_integer_vector(vector):
    d = len(vector)
    denom = list(dict.fromkeys([r.denominator for r in vector]))
    if len(denom) == 1:
        g = 1
    else:
        g = denom[0]
        for i in range(1,len(denom)):
            g = gcd(g,denom[i])
    prod = denom[0]
    for i in range(1,len(denom)):
        prod = prod*denom[i]
    lcm = F(abs(prod),g)
    vector = np.array([(lcm * vector[n]).numerator for n in range(0,d)])
    num = list(dict.fromkeys(vector))
    if len(num) == 1:
        g = 1
    else:
        g = num[0]
        for i in range(1,len(num)):
            g = gcd(g,num[i])
    return F(1,g)*vector

#Basic Gauß-Jordan-Algorithm for inverting a matrix over the rationals
def inverse_matrix(matrix):
    if len(matrix) != len(matrix[0]):
    	return(matrix, ' is not square')
    d = len(matrix)
    pivots = [i for i in range(d)]
    for j in range(d):
        pivot = np.argmax(abs(matrix[j:,j].copy()))
        if matrix[pivot + j,j] == 0:
            print(matrix, ' is singular')
            return 0
        if pivot > 0:
            pivots[j], pivots[pivot+j] = pivots[pivot+j], pivots[j]
            matrix[[j,pivot+j]] = matrix[[pivot+j,j]]
        pivot = F(1,matrix[j,j])
        matrix[j,j] = pivot
        I = [i for i in range(d) if i != j]
        matrix[I,j] = pivot * matrix[I,j]
        matrix[np.ix_(I,I)] = matrix[np.ix_(I,I)] - matrix[I,j].reshape(len(I),1) @ matrix[j,I].reshape(1,len(I))
        matrix[j,I] = -pivot * matrix[j,I]
    perm = np.full(d,F(0,1))
    for i in range(d):
        perm[pivots] = matrix[i,:]
        matrix[i,:] = perm
    return matrix

#Computes the row echolon form of a matrix
#Basicly the same mechanism as the gauß-jordan-algorithm
def echolon_form(matrix):
	row = len(matrix)
	col = len(matrix[0])
	rownumber = 0
	columnnumber = 0
	while rownumber < row and columnnumber < col:
		for i in range(columnnumber,col):
			if any(matrix[rownumber:,i] != np.zeros(row-rownumber)):
				break
		if all(matrix[rownumber:,i] ==	np.zeros(row-rownumber)):
			break	
		if matrix[rownumber][i] != 0:
			matrix[rownumber] /= matrix[rownumber][i]
		else:
			for j in range(rownumber,row):
				if matrix[j][i] != 0:
					break
			matrix[[rownumber,j]] = matrix[[j,rownumber]]
			matrix[rownumber] /= matrix[rownumber][i]
		for j in range(rownumber+1,row):
			matrix[j] -= matrix[j][i] * matrix[rownumber]
		rownumber += 1
		columnnumber = i+1 
	return matrix

#Computes the reduced echolon form of a matrix
def reduced_echolon_form(matrix):
	matrix = echolon_form(matrix)
	row = len(matrix)
	col = len(matrix[0])
	rownumber = row-1
	columnnumber = col-1
	while rownumber > 0 and columnnumber > 0:
		for i in range(rownumber,-1,-1):
			if any(matrix[i,:columnnumber+1] != np.zeros(columnnumber+1)):
				break
		for j in range(columnnumber):
			if matrix[i][j] == 1:
				break
		for n in range(i-1,-1,-1):
			matrix[n] -= matrix[n][j] * matrix[i]
		rownumber = i-1 
		columnnumber = j-1 
	return matrix

#computes rank of matrix
def matrix_rank(matrix):
	row = len(matrix)
	if matrix.shape == (row,):
		if any(matrix != np.zeros(row)):
			return 1
		else:
			return 0
	else:
		matrix = echolon_form(matrix)
		col = len(matrix[0])
		rank = len([i for i in range(row) if any(matrix[i] != np.zeros(col))])
		return rank

#For a set V of vectors given as columns of a matrix, 
#finds all the vectors that are linear dependend of the others. 
#Specifically, the algorithm returns the indices of all vectors in V, 
#such that V without those vectors is linear independent
def find_linear_dependencies(vectors):
	matrix = reduced_echolon_form(vectors.copy())
	row = len(matrix)
	col = len(matrix[0])
	lin_indep = []
	for i in range(row):
		if any(matrix[i] != np.zeros(col)):
			for j in range(col):
				if matrix[i][j] == 1:
					break
			lin_indep.append(j)
	lin_dep = [i for i in range(col) if i not in lin_indep]
	return lin_dep

#Removes previously found linear dependencies in an Array
def remove_linear_dependencies(vectors):
	lin_dep = find_linear_dependencies(vectors.copy())
	return np.delete(vectors,lin_dep,1)


def euclidean_norm(vector):
	sum_of_squares = 0
	for i in range(0,len(vector)):
		sum_of_squares += vector[i] ** 2

	return sqrt(sum_of_squares)


#for an integral square matrix A of full rank,
#the algorithm computes an upper triangular matrix W,
#such that the maximal value in each column is on the diagonal
#and an unimodular matrix U, such that UA = W.
#The algorithm uses the mechanisms of the LLL-algorithm
#to reduce mid-computing entry-explosion.

#Might be updatet to do that for non-square-matrices and not necessarily integer matrices
def hermite_normal_form(matrix, mode = 'integer'):
	if mode not in ['integer', 'fraction']:
		print(mode, ' is not a possible output-mode, please choose either "integer" (by default) or "fraction"')
	d = len(matrix)
	U = np.full([d,d], F(0,1))
	for i in range(d):
		U[i][i] = F(1,1)
	W = matrix.copy()
	D = np.full(d, F(1,1))
	lam = np.full([d,d], F(0,1))
	constants = [3,4]
	for l in range(d):
		if len([*filter(lambda x: x != 0, W[:,l])]) > 0:
			break
	if len([*filter(lambda x: x != 0, W[:,l])]) == 1:		
		if W[:,l].tolist().index(next(filter(lambda x: x != 0, W[:,l]))) == d-1 and W[d-1][l] < 0:
			W[d-1,:] = -W[d-1,:]
			U[d-1][d-1] = -F(-1,1)
	k = 1
	while k <= d-1:
		W,U,lam,col = reduce(W,U,lam,D,k,k-1)
		if col[0] <= min(d-1,col[1]):
			W,U,lam,D = swap(W,U,lam,D,k)
			if k > 1:
				k = k - 1
		elif col[0] == d and col[1] == d and constants[1]*(D[k-2]*D[k] + lam[k][k-1]**2) < constants[0]*D[k-1]**2:
			W,U,lam,D = swap(W,U,lam,D,k)
			if k > 1:
				k = k - 1
		else:
			for i in range(k-2,-1,-1):
				W,U,lam,col = reduce(W,U,lam,D,k,i)
			k = k + 1
	for i in range(floor(d/2)):
		W[[i,d-1-i]] = W[[d-1-i,i]]
		U[[i,d-1-i]] = U[[d-1-i,i]]
	for i in range(d):
		for j in range(d):
			if U[i][j].denominator != 1:
				print('Error: ', U, ' should be integral')
				return 0
			if W[i][j].denominator != 1:
				print('Error: ', W, ' should be integral')
				return 0
	if mode == 'fraction':
		return W,U
	elif mode == 'integer':
		return convert_matrix_integral(W),convert_matrix_integral(U)

def minus(C,n):
	for r in range(1,len(C)):
		for s in range(r):
			if r == n or s == n:
				C[r][s] = -C[r][s]
	return C

def reduce(A,B,C,D,n,m):
	d = len(A)
	col = [0,0]
	if len([*filter(lambda x: x != 0, A[m,:])]) > 0 :
		col[0] = A[m,:].tolist().index(next(filter(lambda x: x != 0, A[m,:])))
		if A[m][col[0]] < 0:
			C = minus(C,m)
			B[m,:] = -B[m,:]
			A[m,:] = -A[m,:]
	else:
		col[0] = d
	if len([*filter(lambda x: x != 0, A[n,:])]) > 0:
		col[1] = A[n,:].tolist().index(next(filter(lambda x: x != 0, A[n,:])))
		if A[n][col[1]] < 0:
			C = minus(C,n)
			B[n,:] = -B[n,:]
			A[n,:] = -A[n,:]
	else:
		col[1] = d
	if col[0] <= d - 1:
		q = floor(A[n][col[0]]/A[m][col[0]])
	else:
		if 2*abs(C[n][m]) > D[m]:
			q = ceil(C[n][m]/D[m])
		else:
			q = 0
	q = F(q,1)		
	if q != 0:
		A[n,:] -= q*A[m,:]
		B[n,:] -= q*B[m,:]
		C[n][m] -= q*D[m]
		for j in range(m):
			C[n][j] -= q*C[m][j]
	return A,B,C,col

def swap(A,B,C,D,n):
	A[[n,n-1]] = A[[n-1,n]]
	B[[n,n-1]] = B[[n-1,n]]
	for j in range(n-1):
		C[n][j],C[n-1][j] = C[n-1][j],C[n][j]
	for i in range(n+1,len(A)):
		t = C[i][n-1]*D[n] - C[i][n]*C[n][n-1]
		C[i][n-1] = F(C[i][n-1]*C[n][n-1]+C[i][n]*D[n-2],D[n-1])
		C[i][n] = F(t,D[n-1])
	D[n-1] = F(D[n-2]*D[n] + C[n][n-1]**2,D[n-1])
	return A,B,C,D