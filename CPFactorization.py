import numpy as np
from cvxpy import Variable, Problem, Minimize
from math import comb
from fractions import Fraction as F
from random import randrange, seed
from CopositivePartition import copositive_partition
from CopositiveMinimum import compute_copositive_minimum
from ContiguousCOPPerfectForm import contiguous_cop_perfect_matrix
from CopositiveTest import copositive_test
from DoubleDescriptionMethod import double_description_method
from RationalMatrixTools import convert_matrix_fractional, matrix_rank

#For a given completely copositive matrix Q which omits a rational cp-factorisation and a given COP-perfect matrix P,
#this function computes a rational cp-factorisation via a simplex-type method
#If Q doesn't omit a rational cp-factorisation, the algorithm might not terminate
def rational_cp_factorisation(matrix,P):
    #Here we check for squareness, symmetry and matching dimensions
    if len(matrix[0,:]) != len(matrix[:,0]):
        print(matrix,' is not square')
        return 0
    if len(P[0,:]) != len(P[:,0]):
        print(P,' is not square')
        return 0
    if len(matrix) != len(P):
        print(matrix, ' and ', P, ' are not of matching dimension')
        return 0    
    d = len(matrix)
    if all((matrix.T != matrix).reshape(d**2,)) == True:
        print(matrix, ' is not symmetric')
        return 0
    if all((P.T != P).reshape(d**2,)) == True:
        print(P, ' is not symmetric')
        return 0
    #We compute the copositive minimum and the minimal vectors of P, if there are too few, the algorithm returns an error
    minC, MinC = compute_copositive_minimum(P,1)
    if rank_MinC(Minc.copy()) != comb(d+1,2):
        print(P, ' is no COP-perfect, since ', MinC, ' has not enough linear independend elements')
        return 0
    seed()
    #To check wether Q is in the voronoi cone of the current COP-perfect matrix P, we solve the linear program
    #                                                                                                          min <Q,B>
    #                                                                                                          s.t v^T B v >= 0
    #If the optimal value is 0, then Q is in the cone and we can compute a rational cp-factorisation
    #If the optimal value is negative, Q is not in the cone and we calculate the next COP-perfect matrix
    Cmatrix,matrixvector = define_linearprogram(matrix.copy(),MinC.copy())
    B = Variable(comb(d+1,2))
    problem = Problem(Minimize((B @ matrixvector).trace()),[Cmatrtix @ B >= 0])
    while problem.solve() < 0:
        #To get the next matrix, we compute the extreme rays of the voronoi cone of P
        #and than choose an extreme ray R with <R,Q> < 0 by a certain pivot rule (here we used "randomised" selection)
        #Then we compute the contiguous COP-perfect matrix along the direction of R
        V = dual_voronoi_cone(MinC.copy())
        Ext = [*filter(lambda R: (R @ matrix).trace() < 0,compute_extreme_rays(V,d))]
        R = Ext[randrange(0,len(Ext))]
        P, MinC = contiguous_cop_perfect_matrix(P,R,MinC)
        Cmatrix,matrixvector = define_linearprogram(matrix,MinC)
        B = cp.Variable(comb(d+1,2))
        problem = Problem(Minimize((B @ matrixvector).trace()),[Cmatrix @ B >= 0])
    #If we are in the voronoi-cone we can compute a rational cp-factorisation by solving the linear program
    #                                                                                                       min 0^T alpha
    #                                                                                                       s.t (vec(v_{1}v_{1}^T),...,vec(v_{n}v_{n}^T))alpha = vec(Q)
    #                                                                                                           alpha >= 0
    #                                                                                                           n = |MinC|, v_{i} is in MinC
    #where vec(A) is the vector that results in stacking the rows of A on top of each other.
    alpha = caratheodory(matrix,MinC)
    print('The CP factorization of ', matrix, ' is:')
    return alpha, MinC


#For a given matrix Q and a COP-perfect matrix P gives either a copositive witness matrix B with <Q,B> < 0 if Q is not completely positive
#or a rational cp-factorization of Q, if it omits one. If it doesn't, the algorithms might not terminate
#The only differences to the rational_cp_factorisation-function are a test wether <P,Q> < 0 for all COP-perfect matrices which are seen in the algorithm
# and a test wether the chosen extremal ray R is nonnegativ ( and as such copositive ). Because R is chosen, such that <R,Q> < 0, P or R are Witness matrices for
#the non-complete positivity of Q
def cp_membership_algorithm(matrix, P):
    if len(matrix[0,:]) != len(matrix[:,0]):
        print(matrix,' is not square')
        return 0
    if len(P[0,:]) != len(P[:,0]):
        print(P,' is not square')
        return 0
    if len(matrix) != len(P):
        print(matrix, ' and ', P, ' are not of matching dimension')
        return 0 
    d = len(matrix)
    if all((matrix.T != matrix).reshape(d**2,)) == True:
        print(matrix, ' is not symmetric')
        return 0
    if all((P.T != P).reshape(d**2,)) == True:
        print(P, ' is not symmetric')
        return 0
    minC, MinC = compute_copositive_minimum(P,1)
    if rank_MinC(Minc.copy()) != comb(d+1,2):
        print(P, ' is no COP-perfect, since ', MinC, ' has not enough linear independend elements')
        return 0
    if (P @ matrix).trace() < 0:
        print(matrix, ' is not completely positive with COP-perfect witness matrix:')
        return P
    seed()
    Cmatrix,matrixvector = define_linearprogram(matrix.copy(),MinC.copy())
    B = Variable(comb(d+1,2))
    problem = Problem(Minimize(B @ matrixvector),[Cmatrix @ B >= 0])
    while problem.solve() < 0:
        V = dual_voronoi_cone(MinC.copy())
        Ext = [*filter(lambda R: (R @ matrix).trace() < 0,compute_extreme_rays(V,d))]
        R = Ext[randrange(0,len(Ext))]
        if all((R >= np.zeros([d,d])).reshape(d**2,)):
            print(matrix, ' is not completely positive with copositive witness matrix:')
            return R
        P, MinC = contiguous_cop_perfect_matrix(P,R,MinC)
        if (P @ matrix).trace() < 0:
            print(matrix, ' is not completely positive with COP-perfect witness matrix:')
            return P
        Cmatrix,matrixvector = define_linearprogram(matrix.copy(),MinC.copy())
        B = Variable(comb(d+1,2))
        problem = Problem(Minimize(B @ matrixvector),[Cmatrix @ B >= 0])
    alpha = caratheodory(matrix.copy(),MinC.copy())
    print(matrix, 'is completely positive with CP factorization:')
    return alpha, MinC

def rank_MinC(MinC):
    return matrix_rank(dual_voronoi_cone(MinC.copy()))

def define_linearprogram(matrix,MinC):
    d = len(matrix)
    for i in range(d):
        for j in range(i+1,d):
            matrix[i][j] = 2*matrix[i][j]
    matrixvector = np.array([matrix[i][j] for i in range(d) for j in range(i,d)])
    Cmatrix = dual_voronoi_cone(MinC)
    return Cmatrix, matrixvector

def dual_voronoi_cone(MinC):
    d = len(MinC[0])
    n = len(MinC)
    V = np.zeros([n,int(comb(d+1,2))])
    for i in range(n):
        xmatrix = MinC[i].reshape(d,1) @ MinC[i].reshape(1,d)
        for j in range(d):
            for k in range(j+1,d):
                xmatrix[j][k] = 2*xmatrix[j][k]
        xvector = np.array([xmatrix[j][k] for j in range(d) for k in range(j,d)])
        V[i,:] = xvector
    return V

#Up to now we compute the extremalrays with the double description method
#This is probably not optimal since we conjecture that the voronoi-cones of the COP-perfect matrices are
#non degenerate in general. 
#In the future we want to also implement a version of the reverse search algorithm by avis, which will
#probably perform better in general in this instance
def extremalrays(V,dimension):
    A = convert_matrix_fractional(V.copy())
    Rays = double_description_method(A.copy())
    Ext = []
    for R in Rays:
        Extray = np.zeros([dimension,dimension])
        k = 0
        for i in range(dimension):
            for j in range(i,dimension):
                Extray[i][j] = R[k]
                Extray[j][i] = R[k]
                k += 1
        Extray = convert_matrix_fractional(Extray)
        Ext.append(Extray)
    return Ext

def caratheodory(matrix,MinC):
    d = len(MinC[0])
    n = len(MinC)
    Cmatrix = np.zeros([comb(d+1,2),n])
    for i in range(n):
        xmatrix = MinC[i].reshape(d,1) @ MinC[i].reshape(1,d)
        for i in range(d):
            for j in range(i+1,d):
                xmatrix[i][j] = 2*xmatrix[i][j]
        xvector = np.array([xmatrix[i][j] for i in range(d) for j in range(i,d)])
        Cmatrix[:,i] = xvector
    for i in range(d):
        for j in range(i+1,d):
            matrix[i][j] = 2*matrix[i][j]
    matrixvector = np.array([matrix[i][j] for i in range(d) for j in range(i,d)])
    alpha = Variable(n)
    problem = Problem(Minimize(np.zeros(n) @ alpha), [Cmatrix @ alpha <= matrixvector, Cmatrix @ alpha >= matrixvector])
    return alpha.value






    







