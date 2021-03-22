import numpy as np
from math import floor, sqrt 
from fractions import Fraction as F
from CopositivePartition import copositive_partition
from CopositiveTest import copositive_test
from RationalMatrixTools import inverse_matrix, hermite_normal_form, scale_to_integer_vector, fractional_part

#For a strictly copostive matrix Q computes the copositive minimum minC of Q and all minimal vectors MinC of Q
#The idea of the algorithm is to first compute a subdivision of cones of the nonnegative orthant of R^d via simplicial partions of the (d-1)-dimensional standard simplex
#Then a modified Fincke-Pohst-Algorithm is applied to each of those cones
def compute_copositive_minimum(matrix, minC = 'None', mode = 'complete'):
    #Testing for "squareness", symmetry and strict copositivity
    if not copositive_test(matrix.copy(),'strict'):
        print(matrix, ' is not strictly copositive')
        return 0,0
    if len(matrix[0,:]) != len(matrix[:,0]):
        print(matrix, ' is not square')
        return 0,0
    d = len(matrix)
    if all((matrix.T != matrix).reshape(d**2,)) == True:
        print(matrix, ' is not symmetric')
        return 0
    if mode not in ['complete', 'partial']:
        print(mode, ' is not a valid computation mode, please use either "complete" (by default) or "partial"')
        return 0,0
    #Calculating the simplicial partition
    P = copositive_partition(matrix.copy())
    #If minC is not supplied by the user, we choose the smallest diagonal entry as our start value
    if minC == 'None':
        minC = min([matrix[n][n] for n in range(d)])
    MinCrat = []
    MinC = []
    if mode == 'complete':
        #Since we consider cones, we can scale the vertices of the simplices in the partition by a positive value
        #The simplicial partition is computed in such a way that all the vertices are rational,
        #therefore we can scale those to be integer 
        for k in range(len(P)):
            for i in range(d):
                P[k][i] = scale_to_integer_vector(P[k][i])
            #For the matrix of integer-scaled vertices V we compute the hermite normal form W aswell as the integer unimodular matrix U
            #such that W = UV
            #This will make the task of searching the cones significantly easier    
            hermit_matrix, transform_matrix = hermite_normal_form(P[k].T,'fraction')
            transform_matrix = inverse_matrix(transform_matrix.copy())
            #Here the minimal vectors and minimum of a cone is computed. We will use that set of minimal vectors and the minimum for the next cone
            #If there are vectors such that the minimum is smaller, it gets updatet appropriately 
            minC, MinCrat = compute_short_vectors_in_cone(transform_matrix,hermit_matrix,matrix,transform_matrix.T @ matrix @ transform_matrix,minC,MinCrat,mode)
        #Here we transform the vectors, for presentation purposes
        for v in MinCrat:
            u = np.array([v[i].numerator for i in range(0,d)])
            MinC.append(u)
        return minC.numerator/minC.denominator,MinC
    #The partial mode is a special mode for the contigous_cop_perfect_form - Function
    #Basicly it either computes the copositive minimum and all minimal vectors, if that copositive minimum is >= 1
    #otherwise it only computes one integer vector v with v^T Q v < 1 
    elif mode == 'partial':
        for k in range(0,len(P)):
            for i in range(0,d):
                P[k][i] = scale_to_integer_vector(P[k][i])    
            hermit_matrix, transform_matrix = hermite_normal_form(P[k].T,'fraction')
            transform_matrix = inverse_matrix(transform_matrix.copy())
            minC, MinCrat = compute_short_vectors_in_cone(transform_matrix,hermit_matrix,matrix,transform_matrix.T @ matrix @ transform_matrix,minC,MinCrat,mode)
            if minC < 1:
                v = np.array([MinCrat[i].numerator for i in range(d)])
                return minC.numerator/minC.denominator , v
        for v in MinCrat:
            u = np.array([v[i].numerator for i in range(d)])
            MinC.append(u)
        return minC.numerator/minC.denominator,MinC

#This is basicly the modified Fincke-Pohst-Algorithm with is used to enumerate the minimal vectors in a given cone using backtracking
def compute_short_vectors_in_cone(transform_matrix,hermit_matrix,matrix,matrix2,minC,MinC,mode):
    d = len(matrix)
    fun = hermit_matrix.T @ matrix2 @ hermit_matrix
    if not all((fun >= np.zeros([d,d])).reshape(d**2,)):
        print('Matrix is not positive in cone')
        return 0,[]
    #Here we initialize all iterables in the algorithm
    #PartialSums[i] is the value of (e_{i+1}v_{i+1} + ... + e_{d}v_{d})^T Q' (e_{i+1}v_{i+1} + ... + e_{d}v_{d})
    #and will be used to compute upper bounds and decide wether a given vector v has the chance to be minimal
    PartialSums = np.full(d,F(0,1))
    #v is the current considered vector in the cone
    v = np.full(d,F(0,1))
    #m[i] are the upper bounds of the value n[i]
    m = [0 for i in range(d)]
    #n gives the value to construct the vector v
    n = [0 for i in range(d)]
    #startvalue[i] is the smallest value of n[i]
    startvalue = [F(0,1) for i in range(0,d)]
    #The vector alpha uses the vector n to construct the vector v
    alpha = np.full(d,F(0,1))
    #Here we compute the inital upper bounds, startvalues etc.
    PartialSums, v, m, n, startvalue, alpha = Update(d,d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,minC) 
    i = d-1                                                                                                    
    while True:
        #if PartialSum[0] = v^T Q' v is bigger than the current minimun we dont need to consider it
        if PartialSums[0] <= minC:
            #Here we transform the vector v into the actual vector, that might be added to MinC
            u = transform_matrix @ v
            #We only consider it, if u != 0. If the u^T Q u is equal to the current minimum we add it to the minimal vectors
            #if it is smaller, we will update the minimum and minimal vectors accordingly
            if any(u != np.zeros(d)) and all(u >= np.zeros(d)):
                val = u @ matrix @ u
                if val == minC:
                    if not any([all(u == MinC[j]) for j in range(len(MinC))]):
                        MinC.append(u.copy())
                elif val < minC:
                    if mode == 'complete':
                        minC = val
                        MinC = [u.copy()]
                        PartialSums = np.full(d,F(0,1))
                        v = np.full(d,F(0,1))
                        m = [0 for i in range(d)]
                        n = [0 for i in range(d)]
                        startvalue = [F(0,1) for i in range(0,d)]
                        alpha = np.full(d,F(0,1))
                        PartialSums, v, m, n, startvalue, alpha = Update(d,d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,minC)
                    elif mode == 'partial':
                        return val, u
        #Here we get the next vector to consider. If there are no more vectors to consider term will be False and the algorithm will terminate
        term, PartialSums, v, m, n, startvalue, alpha = get_next_vector(d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,minC)
        if not term:
            break
    return minC, MinC
#This is a function which is only used for the contiguous_cop_perfect_matrix-function and works basicly like the compute_copositive_minimum-function in the partial mode,
#except that we are computing all integer vectors v with v^T Q v < 1
def enumerate_S(matrix):
    d = len(matrix)
    P = copositive_partition(matrix)
    Srat = []
    for k in range(len(P)):
        for i in range(d):
            P[k][i] = scale_to_integer_vector(P[k][i])
        hermit_matrix, transform_matrix = hermite_normal_form(P[k].T,'fraction')
        transform_matrix = inverse_matrix(transform_matrix.copy())
        Srat = compute_S_vectors(transform_matrix,hermit_matrix,matrix,transform_matrix.T @ matrix @ transform_matrix,Srat)
    S = []
    for s in Srat:
        u = np.array([s[i].numerator for i in range(0,d)])
        S.append(u)
    return S
              
def compute_S_vectors(transform_matrix,hermit_matrix,matrix,matrix2,S):
    d = len(matrix)
    fun = hermit_matrix.T @ matrix2 @ hermit_matrix 
    PartialSums = np.full(d,F(0,1))
    v = np.full(d,F(0,1))
    m = [0 for i in range(d)]
    n = [0 for i in range(d)]
    startvalue = [F(0,1) for i in range(d)]
    alpha = np.full(d,F(0,1))
    PartialSums, v, m, n, startvalue, alpha = Update(d,d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,1) 
    i = d-1                                                                                                    
    while True:
        if PartialSums[0] < 1:
            u = transform_matrix @ v.copy()
            if any(u != np.zeros(d)) and all(u >= np.zeros(d)):
                val = u.copy() @ matrix @ u.copy()
                if val < 1:
                    if not any([all(u ==S[j]) for j in range(len(S))]):
                        S.append(u.copy())
        term, PartialSums, v, m, n, startvalue, alpha = get_next_vector(d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,1)
        if not term:
            break
    return S
#updates all the iterables accordingly
def Update(i,d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,minC):
    for j in range(i-1,-1,-1):
        number = alpha[j+1:d] @ hermit_matrix[j][j+1:d]
        startvalue[j] = F(fractional_part(-number),hermit_matrix[j][j])
        alpha[j] = F(n[j],hermit_matrix[j][j]) + startvalue[j]
        v[j] = number + alpha[j] * hermit_matrix[j][j]
    for j in range(i-1,-1,-1):
        number = 2 * alpha[j+1:d] @ fun[j,j+1:d]
        if j < d-1:
            Sum = PartialSums[j+1]
        else:
            Sum = 0
        PartialSums[j] = alpha[j]**2 * fun[j][j] + alpha[j] * number + Sum
        c =  Sum + startvalue[j] * number + fun[j][j] * startvalue[j]**2
        if c >= minC:
            m[j] = 0
        else:
            m[j] = compute_upper_bound(F(number + 2* startvalue[j] * fun[j][j],hermit_matrix[j][j]), F(fun[j][j],hermit_matrix[j][j]**2), minC - c) 
    return PartialSums,v,m,n,startvalue,alpha
#computes the needed upper bounds for m[i]
def compute_upper_bound(a,b,c):
    a2 = a/b
    c2 = c/b
    x = floor(1/2 * (-a2 + sqrt( a2 ** 2 + 4*c2)))
    while True:
        test1 = (c2 - a2 * x - x**2 >= 0)
        test2 = (c2 - a2 * (x + 1) - (x + 1)**2 >= 0)
        if test1 and not test2:
            break
        if not test1:
            x -=  1
        if test2:
            x += 1        
    return x
#gives us the next vector in the cone which is to be considered. If there are none, returns False
def get_next_vector(d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,minC):
    for i in range(d):
        if n[i] < m[i]:
            n[i] += 1
            for j in range(i):
                n[j] = 0
            PartialSums, v, m, n, startvalue, alpha= Update(i+1,d,fun,hermit_matrix,PartialSums,v,m,n,startvalue,alpha,minC)
            return True,PartialSums,v,m,n,startvalue,alpha
    return False,PartialSums,v,m,n,startvalue,alpha

