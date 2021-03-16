FASTER Simulated Annealing algorithm to solve Travelling Salesman Problem in Python
===================


⚠ What's different about my version vs original is that it doesn't recalculate whole length every time and instead considers the difference. However, would be faster still if reversing was to be done only when actually needed. Also, for an even more optimal solution in case of a large number of nodes, one could store links from nodes to nodes instead of a direct array path.


----------
----------
----------


Using [simulated annealing](https://en.wikipedia.org/wiki/Simulated_annealing) metaheuristic to solve the [travelling salesman problem](https://en.wikipedia.org/wiki/Travelling_salesman_problem), and visualizing the results. 

Starts by using a greedy algorithm (nearest neighbour) to build an initial solution.

A simple implementation which provides decent results.

----------

An example of the resulting route on a TSP with 100 nodes.

![Route Graph](http://i.imgur.com/IY9cCJG.png)

The fitness (objective value) through iterations.

![Learning Plot](http://i.imgur.com/EVOkZs3.png)


----------

**References**
[Kirkpatrick et al. 1983: "Optimization by Simulated Annealing"](http://leonidzhukov.net/hse/2013/stochmod/papers/KirkpatrickGelattVecchi83.pdf)

http://www.blog.pyoung.net/2013/07/26/visualizing-the-traveling-salesman-problem-using-matplotlib-in-python/
