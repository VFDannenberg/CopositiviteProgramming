import numpy as np
from RationalMatrixTools import matrix_rank, remove_linear_dependencies, inverse_matrix, scale_to_integer_vector

#This is a basic implementation of the double description method by Motzkin et. al.
#For a given pointed polyhedral cone given by a matrix A, it computes all the extremal rays of that cone
#We still need to do some work here, to present a complete implementation as well as further improvements in our implementation
def double_description_method(cone,mode = 'algebraic', ins_order = 'lexicographic'):
	#Here we check for pointedness of the cone
	if matrix_rank(cone.copy()) != len(cone[0]):
		print('The cone given by cone = ' , cone , ' is not pointed')
		return 0
	if mode not in ['algebraic', 'combinatorial']:
		print(mode, ' is not a valid adjacency-checking-procedure, please use either "algebraic" (by default) or "combinatorial" (not yet implemented)')
		return 0
	if ins_order not in ['lexicographic', 'max-cut-off', 'min-cut-off', 'max-intersection']:
		print(ins_order, ' is not a valid insertion order, please use either "lexicographic" (by default), "max-cut-off, "min-cut-off" or "max-intersection"')
		return 0
	col = len(cone[0])
	row = len(cone)
	#Firstly we compute a relaxed cone by choosing only certain rows of A, such that 
	#the cone given by that matrix is also pointed, but the extremal rays can be computed easily, by inversion
	#We save the rows we didn't take in the set tau. Over the course of this iteration, we step by step take a 
	#row in tau and add it to the relaxed cone, compute the extremal rays of that cone, and add the next row
	#until we get our initial cone and all extremal rays of it 
	tau = find_inital_pointed_cone(cone)
	taucomp = [i for i in range(row) if i not in tau]
	R = compute_extremal_rays_les(cone,taucomp)
	while len(tau) > 0:
		#Which row we choose can have a huge impact on computation time. In the function get_next_index we
		#implemented four known insertion orders
		i = get_next_index(cone,R,tau,ins_order)
		#Here we update the extreme rays in dependence of the row we choose, obviously if a_{i} is the row, all
		#extremal rays r need to satisfy a_{i}^T r >= 0. If an extreme ray doesnt, we delete it from the list of extreme rays
		#but save it in the Set R-, because we will need it to compute the rest of the extreme rays
		#R1+ and R1- are sets of extreme rays in R+ and R- which satisfy certain support conditions
		#which can be used to quickly check the adjacency of those rays 
		R, R_plus,R_minus,R1_plus,R1_minus = Update(cone,i,tau,R)
		#There are a few efficient adjacency checks which can quickly see wether certain extreme rays are adjacent or not
		#so we will use that here, to get as many as possible as efficient as possible
		R = quick_adjacency_check(cone,tau,R,R1_plus,R_minus)
		#Temp+ and Temp- are the sets R+\R1+ and R-\R1- since we already checked adjacency for those rays
		Temp_plus = [r for r in R_plus if any([all(r == R1_plus[j]) for j in range(len(R1_plus))]) == False] 
		Temp_minus = [r for r in R_minus if any([all(r == R1_minus[j]) for j in range(len(R1_minus))]) == False] 
		#Another quick adjacency check
		R = quick_adjacency_check(cone,tau,R,Temp_plus,R1_minus)
		#Here we basicly check the adjacency of all the remaining rays, and construct the new extreme rays accordingly
		#The adjacency_check-function has two modes "algebraic" which is based on matrix-rank-computation
		#and the not-yet implemented "combinatorial"-mode, which checks the supports of given extreme rays to determine adjacency
		for r in Temp_plus:
			for s in Temp_minus:
				Index = []
				suppr_tau = [i for i in range(row) if cone[i] @ r != 0 and i not in tau]
				supps_tau = [i for i in range(row) if cone[i] @ s != 0 and i not in tau]
				diff = [i for i in suppr_tau if i not in supps_tau]
				for ray in R:
					suppray_tau = [i for i in range(row) if cone[i] @ ray != 0 and i not in tau]
					diffray = [i for i in suppray_tau if i not in suppr_tau]
					if len(diffray) == 1:
						if diffray[0] not in Index:
							Index.append(diff[0])
				intersect = [i for i in supps_tau if i in Index]
				if len(intersect) == 0:
					comp = [i for i in tau if cone[i] @ (r + s) == 0]
					if len(comp) <  col - 2:
						if len(diff) == 1:
							R.append((-cone[i] @ s) * r + (cone[i] @ r) * s)
						elif adjacency_check(cone,r,s,tau,mode):
							R.append((-cone[i] @ s) * r + (cone[i] @ r) * s)
				elif len(diff) == 1:
					R.append((-cone[i] @ s) * r + (cone[i] @ r) * s)
		tau.remove(i)
	Ext = [scale_to_integer_vector(R[i]) for i in range(len(R))]
	return Ext 
#Here we compute the extreme rays of the initial relaxed cone, which is easily done by inverting the given matrix
def compute_extremal_rays_les(cone,index):
	R = inverse_matrix(cone[index].copy())
	Ext = [scale_to_integer_vector(R[:,i]) for i in range(len(R))]
	return Ext

#as explanined here we check for the adjacency of two given extreme rays
def adjacency_check(cone,r,s,tau,mode):
	if mode == 'algebraic':
		row = len(cone)
		col = len(cone[0])
		suppcomp = [i for i in range(row) if cone[i] @ r == 0 and cone[i] @ s == 0]
		index = [i for i in suppcomp if i not in tau]
		if matrix_rank(cone[index]) == col - 2:
			return True
		else:
			return False

def find_inital_pointed_cone(cone):
	row = len(cone)
	col = len(cone[0])
	if row == col:
		return []
	else:
		sigma = []
		rank = 0
		for i in range(row):
			temp = sigma.copy()
			temp.append(i)
			if matrix_rank(cone[temp].copy()) == rank + 1:
				sigma.append(i)
				rank = rank + 1
				if rank == col:
					break
		tau = [i for i in range(row) if i not in sigma]
		return tau
#Here we get the next row according to a given insertion order
#Max-cut-off chooses the row, such that the most extreme rays don't satisfy the condition a_{i}^T r >= 0
#Min-cut-off chooses the row, such that the least extreme rays don't satisfy the condition a_{i}^T r >= 0
#Lexicographic chooses the row whose index is the lexicographically smallest
#Max-Intersection chooses the row, such that the most extreme rays satisfy the condition a_{i}^T == 0

#There isn't really a general rule for which insertion order to choose, it depends on the cone, and can be very non-obvious
def get_next_index(cone,R,tau,ins_order):
	if ins_order == 'max-cut-off':
		biggest = [len([r for r in R if cone[i,:].dot(r) < 0]) for i in tau]
		return tau[argmax(biggest)]
	elif ins_order == 'min-cut-off':
		smallest = [len([r for r in R if cone[i,:].dot(r) < 0]) for i in tau]
		return tau[argmin(smallest)]
	elif ins_order == 'lexicographic':
		return tau[0]
	elif ins_order == 'max-intersection':
		biggest = [len([r for r in R if cone[i,:].dot(r) == 0]) for i in tau]
		return tau[argmax(biggest)]

#Here we update the cone and extremal rays of the new cone in dependence of the chosen new row
def Update(cone,i,tau,R):
	row = len(cone)
	R_plus = [r for r in R if cone[i] @ r > 0 ]
	R_minus = [r for r in R if cone[i] @ r < 0]
	R = [r for r in R if cone[i] @ r >= 0]
	R1_plus = [] 
	for r in R_plus:
		suppcomp = [i for i in range(row) if cone[i] @ r == 0]
		if len([i for i in suppcomp if i not in tau]) == row - 1:
			R1_plus.append(r)
	R1_minus = []
	for r in R_minus:
		suppcomp = [i for i in range(row) if cone[i] @ r == 0]
		if len([i for i in suppcomp if i not in tau]) == row - 1:
			R1_minus.append(r)
	return R,R_plus,R_minus,R1_plus,R1_minus

#Here we carry the aformentioned quick adjacency checks out, which consist basicly in checking if the length of the difference of the supports is equal to 1
def quick_adjacency_check(cone,tau,R,R1,R2):
	row = len(cone)
	for r in R1:
		for s in R2:
			suppr_tau = [i for i in range(row) if cone[i] @ r != 0 and i not in tau]
			supps_tau = [i for i in range(row) if cone[i] @ s != 0 and i not in tau]
			diff = [i for i in suppr_tau if i not in supps_tau]
			if len(diff) == 1:
				R.append((-cone[i] @ s) * r + (cone[i] @ r) * s)
	return R



