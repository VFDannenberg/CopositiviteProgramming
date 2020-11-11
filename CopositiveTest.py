import numpy as np
from cvxpy import Variable, Problem, Maximize, Minimize
from scipy.optimize import minimize, LinearConstraint
from math import sqrt
from scipy.linalg import null_space
from fractions import Fraction as F
from RationalMatrixTools import matrix_rank

#For a given symmetric square matrix Q returns True if the matrix is copositive (respectively strictly copositive),
#or False if the matrix is not copositive (respectively strictly copositive)

#In the future, we will add that if the matrix is not copositive, False and a vector v with v^T Q v <= 0 ( < 0) is returned

def copositive_test(matrix, mode = 'non-strict'):
    #Checking for "squareness" and symmetry
    if len(matrix) != len(matrix[0]):
        print(matrix, ' is not square')
        return 0
    d = len(matrix)
    if all((matrix.T != matrix).reshape(d**2,)) == True:
        print(matrix, ' is not symmetric')
        return 0
    if mode not in ['strict', 'non-strict']:
        print(mode, ' is not a valid test mode, please specify either "non-strict" (by default) or "strict"')
    
    if mode == 'non-strict':
        #This is the pre-processing section.
        #Here we check if the matrix is either obviously copositive ( if it is nonnegative) or non-copositive,
        #by doing very quick checks, that immediately disqualify Q
        if all((matrix >= np.zeros([d,d])).reshape(d**2,)):
            return True
        # since the all-one-vector is non-negative, this is obvious
        if np.ones(d) @ matrix @ np.ones(d) < 0:
            return False    
        for i in range(d):
            #if a diagonal entry is negative, then e_{i}^T Q e_{i} = Q[i][i] < 0
            if matrix[i][i] < 0:
                return False
            #if a diagonal entry is zero and there is a negative term in the respective column/row
            #then we can also construct a nonnegative vector v with v^T Q v < 0
            if matrix[i][i] == 0:
                for j in range(i+1,d):
                    if matrix[i][j] < 0:
                        return False
            #Here we examine all 2x2 blocks of Q with a negative nondiagonal-entry by doing
            #a quick copositive check. If a block is not copositive, Q can not be copositive
            for j in range(i+1,d):
                if matrix[i][j] < 0 and matrix[i][j]**2 > matrix[i][i] * matrix[j][j]:
                    return False
                else if d == 2:
                    reutrn True
            #If there is a whole column/row of Q that is nonnegative, we can "delete" it, because then
            #the copositivity of is not dependend on that column/row. 
            #This is important because it can reduce the size of the matrix significantly for the next test
            if all(matrix[i,:] >= np.zeros(d)):
                minor = np.delete(np.delete(matrix,i, axis = 0), i, axis = 1)
                return copositive_test(minor,mode)
        return gaddum_criterium(matrix,mode)
    elif mode == 'strict':
        #Most of the techniques are the same as in the non-strict case, adaquatly adapted for strict copositivity
        if all((matrix >= np.full([d,d], F(0,1))).reshape(d**2,)) and matrix_rank(matrix) == d:
            return True
        if np.ones(d) @ matrix @ np.ones(d) <= 0:
            return False
        for i in range(d):
            if matrix[i][i] <= 0:
                return False
            for j in range(i+1,d):
                if matrix[i][j] < 0 and matrix[i][j]**2 >= matrix[i][i] * matrix[j][j]:
                    return False
                else if d == 2:
                    return True
            if all(matrix[i,:] >= np.zeros(d)):
                minor = np.delete(np.delete(matrix,i, axis = 0), i, axis = 1)
                return copositive_test(minor,mode)
        #Here we check if there are any nonnegativ vectors in the kernel of Q. If there is such a vector v
        #then obviously v^T Q v = 0, and Q is not strictly copositive       
        if type(matrix[0][0]) == F:
            matrixfloat = np.zeros([d,d])
            for i in range(d):
                matrixfloat[i][i] = float(matrix[i][i])
                for j in range(i+1,d):
                    matrixfloat[i][j] = float(matrix[i][j])
                    matrixfloat[j][i] = matrixfloat[i][j]
            ns = null_space(matrixfloat)
            for i in range(ns.shape[1]):
                if all(np.sign(ns[0,i]) * ns[:,i] >= np.zeros(d)):
                    return False
        else:
            ns = null_space(matrix)
            for i in range(ns.shape[1]):
                if all(np.sign(ns[0,i]) * ns[:,i] >= np.zeros(d)):
                    return False   
        #We still need to add size-reduction techniques here...       
        return gaddum_criterium(matrix, mode)

def gaddum_criterium(matrix,mode):
    #Gaddums Test first checks recursively if all the principal (d-1)x(d-1) minors of Q are copositive (strictly copositive)
    #If there is one that is not, Q is not copositive (strictly copositive)
    #Note that we don't need to check 2x2 principal minors here, because we already done that in the preprocessing
    #After that we solve the linear program
    #                           min y_{1} - y_{2}
    #                           s.t x_{1} + ... + x_{d} = 1
    #                               Qx - y_{1}e + y_{2}e <= 0
    #                               x,y_{1},y_{2} >= 0.
    #If the optimal value of that LP is nonnegative (positive) Q is copositive (strictly copositive)
    d = len(matrix)
    if mode == 'non-strict':
        for i in range(d):
            minor = np.delete(np.delete(matrix,i, axis = 0), i, axis = 1)
            if len(minor) == 2:
                continue
            elif not gaddum_criterium(minor,mode):
                return False   
        matrix_eq = np.ones(d+2)
        matrix_eq[d] = 0
        matrix_eq[d+1] = 0
        func = np.eye(1,d+2,d).reshape(d+2,)
        func[d+1] = -1
        x = Variable(d+2, nonneg = True)
        problem = Problem(Minimize(func @ x), [matrix_eq @ x == 1, np.c_[matrix,np.full([d,1], F(-1,1)), np.full([d,1], F(1,1))] @ x <= np.zeros(d)])
        if problem.solve() >= 0:
            return True
        else:
            return False
    elif mode == 'strict':
        for i in range(d):
            minor = np.delete(np.delete(matrix,i, axis = 0), i, axis = 1)
            if len(minor) == 2:
                continue
            elif not gaddum_criterium(minor,mode):
                return False
        matrix_eq = np.ones(d+2)
        matrix_eq[d] = 0
        matrix_eq[d+1] = 0
        func = np.eye(1,d+2,d).reshape(d+2,)
        func[d+1] = -1
        x = Variable(d+2, nonneg = True)
        problem = Problem(Minimize(func @ x), [matrix_eq @ x == 1, np.c_[matrix,np.full([d,1], F(-1,1)), np.full([d,1], F(1,1))] @ x <= np.zeros(d)])
        if problem.solve() > 0:
            return True
        else:
            return False




#Still in development
#def xvz_milp_copositive_test(matrix, mode = 'non-strict'):
#    if mode not in ['non-strict', 'strict']:
#        print(mode, ' is not a valid test mode, please choose either "non-strict" (by default) or "strict"')
#        return 0
#    d = len(matrix)
#    x = Variable(d)
#    y = Variable(d,boolean = True)
#    v = Variable(d)
#    u = Variable(1)
#    zeros = np.zeros(d)
#    ones = np.ones(d)
#    matrixabs = np.absolute(matrix.copy())
#    maxele = np.amax(matrixabs)
#    problem = Problem(Minimize(-u), [matrix @ x + u * ones - v == 0, ones @ x == 1, x >= zeros, v >= zeros, y >= x, v <= 2*d*maxele*(ones - y)])
#    if mode == 'non-strict':
#        if problem.solve() >= 0:
#            return True
#        else:
#            return False
#    elif mode == 'strict':
#        if problem.solve() > 0:
#            return True
#        else:
#            return False

#def anstreicher_milp_copositive_test(matrix):
#    d = len(matrix)
#    x = Variable(d)
#    y = Variable(d,boolean = True)
#    c = Variable(1)
#    zeros = np.zeros(d)
#    ones = np.ones(d)
#    m = []
#    problem = Problem(Maximize(c), [])
#    if problem.solve() == 0:
#        return True
#    else:
#        return False



