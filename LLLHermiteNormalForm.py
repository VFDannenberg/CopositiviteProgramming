import numpy as np 
from math import ceil, floor
from fractions import Fraction as F

def minus(C,n):
	for r in range(1,len(C)):
		for s in range(0,r):
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
		A[n,:] = A[n,:] - q*A[m,:]
		B[n,:] = B[n,:] - q*B[m,:]
		C[n][m] = C[n][m] - q*D[m]
		for j in range(0,m):
			C[n][j] = C[n][j] - q*C[m][j]
	return A,B,C,col

def swap(A,B,C,D,n):
	A[[n,n-1]] = A[[n-1,n]]
	B[[n,n-1]] = B[[n-1,n]]
	for j in range(0,n-1):
		C[n][j],C[n-1][j] = C[n-1][j],C[n][j]
	for i in range(n+1,len(A)):
		t = C[i][n-1]*D[n] - C[i][n]*C[n][n-1]
		C[i][n-1] = F(C[i][n-1]*C[n][n-1]+C[i][n]*D[n-2],D[n-1])
		C[i][n] = F(t,D[n-1])
	D[n-1] = F(D[n-2]*D[n] + C[n][n-1]**2,D[n-1])
	return A,B,C,D

def hermite_normal_form(A):
	d = len(A)
	U = np.full([d,d], F(0,1))
	for i in range(0,d):
		U[i][i] = F(1,1)
	W = A.copy()
	D = np.full(d, F(1,1))
	lam = np.full([d,d], F(0,1))
	constants = [3,4]
	for l in range(0,d):
		if len([*filter(lambda x: x != 0, A[:,l])]) > 0:
			break
	if len([*filter(lambda x: x != 0, A[:,l])]) == 1:		
		if A[:,l].tolist().index(next(filter(lambda x: x != 0, A[:,l]))) == d-1 and A[d-1][l] < 0:
			A[d-1,:] = -A[d-1,:]
			B[d-1][d-1] = -1
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
	for i in range(0,floor(d/2)):
		W[[i,d-1-i]] = W[[d-1-i,i]]
		U[[i,d-1-i]] = U[[d-1-i,i]]
	Unorm = np.zeros([d,d])
	for i in range(0,d):
		for j in range(0,d):
			Unorm[i][j] = U[i][j].numerator
	return W,Unorm




	


