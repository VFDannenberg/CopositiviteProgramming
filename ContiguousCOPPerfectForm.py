import numpy as np
from fractions import Fraction as F
from CopositiveTest import copositive_test
from CopositiveMinimum import compute_copositive_minimum, enumerate_S

#For a given COP-perfect matrix P and a non-copositve extreme ray R of (V(P))*, the algorithm find the contiguous COP-perfect matrix
#with minC = 1 in the direction of the extreme ray and its minial vectors
def contiguous_cop_perfect_matrix(perfect_matrix,extreme_ray,MinC,method = 'None'):
    if copositive_test(extreme_ray):
        print('The extreme ray', extreme_ray, 'is copositive. Algorithm will not terminate')
    if method not in ['None', 'direct','iterative','combined']:
        print(method, ' method for enumerating S does not exist, please use an existing method ("None" (by default, results in a dynamic choice of method), "direct", "iterative" or "combined" (not yet implemented))')
        return 0
    #The contiguous COP-perfect matrix is of the form P + a*R.
    #First we compute an upper bound u and a lower bound l, such that a is in the interval [l,u],
    #the matrix P+l*R has a copositive minimum of 1 and the matrix P + u*R has a copositive minimum strictly smaller than 1, but is still strictly copositive
    #If by any chance during this iteration process u = a, the matrix P+ u*R with its minimal vectors is returned
    l,u = F(0,1),F(1,1)
    res = copositive_test(perfect_matrix + u*extreme_ray,'strict')
    if res:
        m, M = compute_copositive_minimum(perfect_matrix + u*extreme_ray, 1, mode = 'partial')
        if m == 1:
            if len([*filter(lambda x: any([all(x == MinC[i]) for i in range(0,len(MinC))]) == False,M)]) > 0:
                return perfect_matrix + u*extreme_ray, M
    else: 
        m = 1
    while not res or m == 1:
        if not res:
            u = F(l + u,2)
            res = copositive_test(perfect_matrix + u*extreme_ray,'strict')
            if res:
                m,M = compute_copositive_minimum(perfect_matrix + u*extreme_ray, 1, mode = 'partial')
                if m == 1:
                    if len([*filter(lambda x: any([all(x == MinC[i]) for i in range(0,len(MinC))]) == False,M)]) > 0:
                        return perfect_matrix + u*extreme_ray, M
                elif m < 1:
                    break
        else:
            m, M = compute_copositive_minimum(perfect_matrix + u*extreme_ray, 1, mode = 'partial')
            if m == 1:
                if len([*filter(lambda x: any([all(x == MinC[i]) for i in range(0,len(MinC))]) == False,M)]) > 0:
                    return perfect_matrix + u*extreme_ray, M
            elif m < 1:
                break
            l,u = u,2*u
            res = copositive_test(perfect_matrix + u*extreme_ray,'strict')
    #After computing l and u we will compute the Set S of all nonnegative integer vectors v, such that v^T (P + uR) v < 1
    #Then we will set a = min (1 - v^T P v)/ v^T R v.
    #This can be done with a direct approach where all vectors in S are enumerated and the minimum is calculated.
    #If the copositive minimum of P + u*R is small, this set can get too large for enumeration
    #Therefore in the other approach we will only calculate one vector v in S at the time and reduce u by setting u = (1 - v^T P v)/ v^T R v
    #and repeating this process, until minC P + u*R = 1, at which point u = a
    if method == 'None':
        if m <= 1/2:
            method = 'iterative'
        else:
            method = 'direct'
    if method == 'direct':
        S = enumerate_S(perfect_matrix + u*extreme_ray)
        u = min([F(1 - v @ (perfect_matrix @ v), v @ (extreme_ray @ v)) for v in S])
        MinCnew = []
        for v in MinC:
            if v @ (extreme_ray @ v) == 0:
                MinCnew.append(v)
        for s in S:
            if F(1 - s @ perfect_matrix @ s, s @ extreme_ray @ s) == u:
                MinCnew.append(s)
        return perfect_matrix + u*extreme_ray, MinCnew
    elif method == 'iterative':
        while m < 1:
            v = M[0]
            u = F(1 - v @ (perfect_matrix @ v), v @(extreme_ray @ v))
            m, M = compute_copositive_minimum(perfect_matrix + u*extreme_ray, 1, mode = 'partial')
        return perfect_matrix + u*extreme_ray, M
   # elif method == 'combined':
   #     while m <= 1/2:
   #         v = M[0]
   #         u = F(1 - v @ (perfect_matrix @ v), v @(extreme_ray @ v))
   #         m, M = compute_copositive_minimum(perfect_matrix + u*extreme_ray, 1, mode = 'partial')
   #     S = enumerate_S(perfect_matrix + u*extreme_ray)
   #     u = min([F(1 - v @ (perfect_matrix @ v), v @ (extreme_ray @ v)) for v in S])
   #     for v in MinC:
   #         if v @ (extreme_ray @ v) != 0:
   #             MinC.remove(v)
   #     for s in S:
   #         if s @ ((perfect_matrix + u*extreme_ray )@ s) == u:
   #             MinC.append(s)
   #     return perfect_matrix + u*extreme_ray, MinC