We present a basic python implementation of the simplex algorithm for rational CP-Factorizations by Dutour Sikiric, Schürmann and Vallentin [4], together
with a possible algorithm for solving the CP-Membership-Problem [2][4], which is based on the aformentioned.

Our Implementation is made in a way that it works with Inputs of rational matrices as numpy.arrays with Fraction-Entrys, from Python's fractions-Package.
To use this implementation, one will need the following (external) packages: cvxpy, numpy,  scipy, aswell as the already in python included packages  fractions, math and random.

CopositiveTest contains a function to test wether a given symmetric dxd matrix is copositive ( strictly copositive ) or not and is based on the criterium of Gaddum [5]. Also it uses various known and simple preprocessing techniques.

CopositivePartition contains a function to compute a simplicial partition of the (d-1)-standard-simplex for a given strictly copositive matrix Q, such that v_{i}^T Qv_{j} >= 0 for all vertices of the partition. It is based on the algorithm presented in [1].

RationalMatrixTools contains various functions to handle basic linear algebraic task(rank computations, integer scaling, computing the inverse of a matrix, finding linear dependencies, computing the euclidean norm) for input matrices/vectors with fractional entries. Also it contains a function to compute the hermite normal form of a square regular integer matrix, based on [6]. It will be updated in the future to handle non-square and singular matrices 

DoubleDescriptionMethod contains a very basic implementation of the double description method by Motzkin et. al, as presented in [7]. There is still work to be done there. One could also probably use the double description method python package as found in [9]. 

CopositiveMinimum contains a function to compute the copositive minimum and the minimal vectors of a strictly copositive matrix as introduced and developed in [4]. Our Implementation is based upon the C-Implementation of Dutour Sikiric in [3]. It also contains a function to enumerate the set S in ContiguousCOPPerfectForm.py.

CopositiveMinimumAlternative contains a experimental method of computing the copositive minimum and minimal vectors as presented in [8], which still needs to be delevoped theoretically.

ContiguousCOPPerfectForm contains a function to compute the contiguous COP-perfect matrix (see also [4]) in the direction of an extremal ray of the voronoi cone of the given matrix P based on the approach presented in [4]. It is mainly used in the functions contained in CPFactorization.py.

CPFactorizations contains two functions. One the simplex-like algorithm to compute a rational cp-factorization of a completely positive matrix Q, which omits a rational cp-factorization, see [4]. If Q doesn't have one, the algorithm might not terminate. The second function is a procedure to solve the CP-Membership-Problem. We recently proved that it does so, if the input-matrix is of size 2, and we have good reasons to believe that is also does for general input matrices Q, see also [4]. 



We will develop and improve these implementations in the future, so tips and comments are greatly appreciated

----------------

[1] S. Bundfuss and M. Dür -- Algorithmic copositivity detection by simplicial partition, Linear Algebra and its Applicaitons 428, p. 1511- 1532, 2008

[2] P.J.C. Dickinson and L. Gijben -- On the computational complexity of membership problems for the completely positive cone and its dual, Computional Optimization and Applications 57, p. 403-415, 2014

[3] M. Dutour Sikiric -- Copositive, https://github.com/MathieuDutSik/polyhedral_common, 2018

[4] M. Dutour Sikiric, A.Schürmann and F.Vallentin -- A simplex algorithm for rational CP-Factorization -- Mathematical Programming, 2020

[5] J.W. Gaddum -- Linear inequalities and quadratic forms, Pacific Journal of Mathematics 8, p. 411-414, 1958

[6] G. Havas, B.S. Majewski and K.R. Matthews -- Extended GCD and Hermite Normal Form Algorithms via Lattice Basis Reduction, Experimental Mathematics 7, p. 125-136, 1998

[7] P.N. Malkin -- Computing Markov bases, Gröbner bases, and extrem rays, PhD thesis Université catholique de Louvain, 2007

[8] A. Schürmann -- Computing the copositive minimum, In preperation 

[9] https://github.com/cddlib/cddlib
