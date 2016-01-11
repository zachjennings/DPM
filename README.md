Package for python implementation of a Dirichlet process mixture.

Tries to replicate functionality of the 

Note: currently broken! Algorithm is somewhat implemented but contains errors, in particular covariance matrix inference will be wrong (although may not change final answer much).

Also, using cluster-by-cluster covariance matrices is is currently unusably slow, needs to be optimized (or perhaps implement Neal 2000 (?) algorithm).