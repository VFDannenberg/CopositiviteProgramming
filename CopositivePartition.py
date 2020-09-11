import numpy as np
from CopositiveTest import copositive_test
from fractions import Fraction as F

def simplex_bisection(A,Delta,Fun,i,j,d):
    lam = max(F(Fun[i][j],Fun[i][j]-Fun[i][i]), min(F(Fun[j][j]-Fun[i][j],Fun[i][i]+Fun[j][j]-2*Fun[i][j]),F(Fun[j][j],Fun[j][j]-Fun[i][j])))
    v = lam*Delta[i] + (1-lam)*Delta[j]
    Delta1, Delta2 = Delta.copy(),Delta.copy()
    Delta1[i] = v
    Delta2[j] = v
    Fun1,Fun2 = Fun.copy(),Fun.copy()
    for k in range(0,d):
        Fun1[i][k] = Delta1[k] @ A @ Delta1[i]
        Fun1[k][i] = Fun1[i][k]
        Fun2[j][k] = Delta2[k] @ A @ Delta2[j]
        Fun2[k][j] = Fun2[j][k]
    return [Delta1,Fun1], [Delta2,Fun2]

def copositive_partition(A):
    #if copositive_test(A,'strict') == 'not strictly copositive':
    #    print(A, ' is not in the interior of COP')
    #    return 0
    if len(A[0,:]) != len(A[:,0]):
        print('Matrix is not square')
        return 0
    d = len(A)
    Delta = np.full([d,d],F(0,1))
    
    for i in range(0,d):
        Delta[i][i] = F(1,1)
    P = [[Delta, A.copy()]]
    Simplexpartition = []
    while len(P) != 0:										
        Delta = P[0]	
        i,j = np.unravel_index(np.argmin(Delta[1],axis = None), Delta[1].shape)																						               							               											
        if Delta[1][i][j] >= 0:			
            P.remove(Delta)									
            Simplexpartition.append(Delta[0])							
        else:
            P.remove(Delta)									
            Delta1, Delta2 = simplex_bisection(A,Delta[0],Delta[1],i,j,d)			
            P.append(Delta1)								
            P.append(Delta2)								  
    return Simplexpartition