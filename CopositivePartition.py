import numpy as np
from fractions import Fraction as F
from RationalMatrixTools import euclidean_norm
from CopositiveTest import copositive_test
from math import sqrt

#For a given symmetric dxd strictly copositive matrix Q returns a partition of
#(d-1)-dimensional standard simplex, such that v_{i}^{T}Qv_{j} >= 0 for every pair
#of vertices of the partition.
def copositive_partition(matrix):
    #Checking for "squareness", symmetry and strict copositivity
    if len(matrix[0,:]) != len(matrix[:,0]):
        print(matrix,' is not square')
        return 0
    if matrix.T != matrix:
        print(matrix,' is not symmetric')
    if not copositive_test(matrix,'strict'):
        print('Warning:', matrix, ' is not strictly copositive. Algorithm might not terminate\n')
    d = len(matrix)
    #Initializing the (d-1)-dimensional standard simplex as a matrix of its vertices e_1,...,e_d
    Delta = np.full([d,d],F(0,1))
    for i in range(d):
        Delta[i][i] = F(1,1)
    #We store the simplices of the partition as a pair of matrices, with the first one representing the vertices
    #of the simplex, and the second one storing all values v_{i}^TQv_{j} for pairs of vertices of the simplex
    #"Negative Edges" ( edges where v_{i}^TQv_{j} >= 0 is not satisfied) can then easily be found
    P = [[Delta, matrix.copy()]]
    Simplexpartition = []
    while len(P) != 0:
            #We select the simplices we bisect in a "Breath-First"-Manner                  
            Delta, Fun = P.pop(0)
            #Finding a negative edge of the selected simplex   
            i,j = np.unravel_index(np.argmin(Fun,axis = None), Fun.shape)
            #If there are no negative edges, then we can add the simplex to the final partition,
            #otherwise we bisect it on the found negative edge                                                                                                                                                                                       
            if Fun[i][j] >= 0:                                              
                Simplexpartition.append(Delta)                          
            else:                               
                Delta1, Delta2 = simplex_bisection(matrix,Delta,Fun,i,j,d)          
                P.append(Delta1)                                
                P.append(Delta2)                                  
    return Simplexpartition
#Is still in development
def copositive_partition_longest_edge(matrix):
    d = len(matrix)
    Delta = np.full([d,d],F(0,1))
    for i in range(0,d):
        Delta[i][i] = F(1,1)
    Norms = np.full([d,d],sqrt(2))
    for i in range(d):
        Norms[i][i] = 0
    P = [[Delta,matrix.copy(),Norms]]
    while exists_negative_edge(P):
        k,i,j = get_longest_edge(P)
        Delta,Fun,Norms = P.pop(k)
        Delta1, Delta2 = Delta.copy(),Delta.copy()
        Fun1,Fun2 = Fun.copy(),Fun.copy()
        Norms1,Norms2 = Norms.copy(),Norms.copy()
        v = F(1,2)*Delta[i] + F(1,2)*Delta[j]
        Delta1[i] = v
        Delta2[j] = v
        for l in range(d):
            Fun1[i][l] = Delta1[l] @ matrix @ Delta1[i]
            Norms1[i][l] = euclidean_norm(Delta1[l] - Delta1[i])
            Fun2[j][l] = Delta2[l] @ matrix @ Delta2[j]
            Norms2[j][l] = euclidean_norm(Delta2[l] - Delta2[j])
            if l != i:
                Norms1[l][i] = Norms1[i][l]
                Fun1[l][i] = Fun1[i][l]
            if l != j:
                Fun2[l][j] = Fun2[j][l]
                Norms2[l][j] = Norms[j][l]
        P.append([Delta1,Fun1,Norms1])
        P.append([Delta2,Fun2,Norms2])
    Simplexpartition = []
    for Delta in P:
        Simplexpartition.append(Delta[0])
    return Simplexpartition
#Is still in development
def simultanious_partition(matrix):
    if len(matrix[0,:]) != len(matrix[:,0]):
        print('Matrix is not square')
        return 0
    d = len(matrix)
    Delta = np.full([d,d],F(0,1))
    for i in range(0,d):
       Delta[i][i] = F(1,1)
    P = [[Delta, matrix.copy()]]
    Simplexpartition = []
    while len(P) != 0:
        Delta, Fun = P[0]    
        i,j = np.unravel_index(np.argmin(Fun,axis = None), Fun.shape)                                                                                                                                                                                     
        if Fun[i][j] >= 0:         
            P.pop(0)                                 
            Simplexpartition.append(Delta)
        else:
            v = F(1,2)*(Delta[i] + Delta[j])
            partitions = []
            for k in range(len(P)):
                edge,n,m = is_edge(Delta[i],Delta[j],P[k][0])
                if edge:
                    Delta1, Delta2 = P[k][0].copy(),P[k][0].copy()
                    Delta1[n] = v
                    Delta2[m] = v
                    Fun1,Fun2 = P[k][1].copy(),P[k][1].copy()
                    for l in range(0,d):
                        Fun1[n][l] = Delta1[l] @ matrix @ Delta1[n]
                        Fun1[l][n] = Fun1[n][l]
                        Fun2[m][l] = Delta2[l] @ matrix @ Delta2[m]
                        Fun2[l][m] = Fun2[m][l]
                    P.append([Delta1,Fun1])
                    P.append([Delta1,Fun1])
                partitions.append(k)
            i = 0    
            for k in partitions:
                P.pop(k-i)
                i += 1
    return Simplexpartition

#For a given edge of a simplex returns two simplices that are the bisektion on the
#middle point of the givin edge
def simplex_bisection(matrix,Delta,Fun,i,j,d):
    #v represents the midpoint of the given edge
    v = F(1,2)*(Delta[i] + Delta[j])
    #Initializing the bisected simplices
    Delta1, Delta2 = Delta.copy(),Delta.copy()
    Delta1[i],Delta2[j] = v,v
    Fun1,Fun2 = Fun.copy(),Fun.copy()
    #Here we update the functions, which is basicly updating only one column or row
    #of the value-matrix by simple calculation
    for k in range(d):
        Fun1[i][k] = Delta1[k] @ matrix @ Delta1[i]
        Fun1[k][i] = Fun1[i][k]
        Fun2[j][k] = Delta2[k] @ matrix @ Delta2[j]
        Fun2[k][j] = Fun2[j][k]
    return [Delta1,Fun1], [Delta2,Fun2]

#Is still in development
def exists_negative_edge(P):
    for Delta in P:
        if len((Delta[1])[Delta[1] < 0]) > 0:
            return True
    return False

#Is still in development
def get_longest_edge(P):
    length = []
    index = []
    for l in range(0,len(P)):
        Norms = P[l][2]
        i,j = np.unravel_index(np.argmax(Norms,axis = None), Norms.shape)
        length.append(Norms[i][j])
        index.append([l,i,j])
    k = np.argmax(length)
    return index[k]
    
#Is still in development
def is_edge(v,w,Delta):
    d = len(Delta)
    for i in range(0,d):
        if all(v == Delta[i]):
            for j in range(0,d):
                if all(w == Delta[j]):
                    return True,i,j
    return False,0,0
