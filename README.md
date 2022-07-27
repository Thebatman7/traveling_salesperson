# traveling_salesperson
The framework in this repository includes a user interface that generates a specified number of random points (the
cities) on a 2-D canvas. The problems are generated based on a random seed, which you can set for debugging
purposes. 
Code implements a branch and bound algorithm to find solutions to the traveling salesperson problem (TSP).
A branch-and-bound algorithm consists of a systematic enumeration of candidate solutions by means of state space search: 
the set of candidate solutions is thought of as forming a rooted tree with the full set at the root. The algorithm explores
branches of this tree, which represent subsets of the solution set. Before enumerating the candidate solutions of a branch, 
the branch is checked against upper and lower estimated bounds on the optimal solution, and is discarded if it cannot produce
a better solution than the best one found so far by the algorithm.
The algorithm depends on efficient estimation of the lower and upper bounds of regions/branches of the search space. 
If no bounds are available, the algorithm degenerates to an exhaustive search.
