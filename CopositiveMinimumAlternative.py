import numpy as np
from math import ceil
from fractions import Fraction as F 
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from CopositiveTest import copositive_test

#This is still an experimental method, which doesn't really work right now.
#It also uses the idea of the Fincke-Pohst-Algorithm. Howerever, unlike the classic method, 
#here we don't compute a subdivision of the nonnegative Orthant via simplicial partition.
#Instead we compute upper bounds of the value of v_{i} by solving the (in general non-convex) QCQP
#                                                                           max x_{i}
#                                                                           s.t x^T Q x <= minC
#                                                                               x_{j} = v_{j}, j < i
#                                                                               x >= 0
#It still needs to be developed how we can approximately solve this QCQP efficiently and good

def compute_copositive_minimum_qcqp(matrix,mode = 'complete'):
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
    #Again we initialize the iterables, as in the classic method.
    m = [0 for n in range(d)]
    v = np.zeros(d)
    PartialSums = np.full(d,F(0,1))
    minC = min([matrix[n][n] for n in range(d)])
    MinC = [np.eye(1,d,n).reshape(d,) for n in range(d) if matrix[n][n] == minC]
    m, PartialSums = update(0,d,matrix,m,v,PartialSums,minC,MinC)
    while True:
        #These steps are basicly the same as in the classic method
        if PartialSums[d-1]  <= minC:
            if any(v != np.zeros(d)) and all(v >= np.zeros(d)):
                val = v @ A @ v
                if val == minC:
                    if not any([all(u == MinCrat[j]) for j in range(len(MinC))]):
                        MinCrat.append(v.copy())
                elif val < minC:
                    if mode == 'complete':
                        minC = val
                        MinCrat = [v.copy()]
                        m = [0 for n in range(d)]
                        v = np.zeros(d)
                        PartialSums = np.full([d,d],F(0,1))
                        m, PartialSums = update(0,d,matrix,m,v,PartialSums,minC,MinC)
                elif mode == 'partial':
                    return val,v
        term, m, v, PartialSums = get_next_vector(d,matrix,m,v,PartialSums,minC,MinC)
        if not term:
            break
    return minC, MinC 
#Basicly the same as in the classic method
def update(i,d,matrix,m,v,PartialSums,minC,MinC):
    for j in range(i+1,d):
        if j > 0:
            Sum = PartialSums[j-1]
        else:
            Sum = 0
        PartialSums[j] = v[j]**2 * matrix[j][j] + 2*v[j]* (v[0:j] @ matrix[0:j,j]) + Sum
        if PartialSums[j] >= minC:
            m[j] = 0
        else:
            m[j] = compute_upper_bound_qcqp(j,d,matrix,v,minC,MinC)
    return m, PartialSums
#Here we solve the aforementioned (in general non-convex) QCQP.
#We might experiment with different solvers for non-convex QCQPs here. 
def compute_upper_bound_qcqp(i,d,matrix,v,minC,MinC):
    NLC = NonlinearConstraint(lambda x: x @ matrix @ x, 0, minC)
    LCI = LinearConstraint(np.eye(d),0,np.inf)
    if i == 0:
        return minimize(lambda x: -np.eye(1,d,i) @ x,constraints = [NLC,LCI]).fun
    else:
        LCE = LinearConstraint(np.eye(i,d,0),v[0:i],v[0:i])
        return minimize(lambda x: -np.eye(1,d,i).reshape(d,) @ x,MinC[len(MinC)-1] + np.ones(d),constraints = [NLC,LCI,LCE]).fun

#Basicly the same as in the classic method
def get_next_vector(d,matrix,m,v,PartialSums,minC,MinC):
    for i in range(d-1,-1,-1):
        if v[i] < m[i]:
            v[i] += 1
            for j in range(i+1,d):
                v[i] = 0
            m, PartialSums = update(i-1,d,matrix,m,v,PartialSums,minC,MinC)
            return True, m, v, PartialSums
    return False, m, v, PartialSums





