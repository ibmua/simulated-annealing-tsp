import math
import random
import visualize_tsp
import matplotlib.pyplot as plt


class SimAnneal(object):
    def __init__(self, coords, T=-1, alpha=-1, stopping_T=-1, stopping_iter=-1):
        self.coords = coords
        self.N = len(coords)
        self.T = math.sqrt(self.N) if T == -1 else T
        self.T_save = self.T  # save inital T to reset if batch annealing is used
        self.alpha = 0.995 if alpha == -1 else alpha
        self.stopping_temperature = 1e-8 if stopping_T == -1 else stopping_T
        self.stopping_iter = 100000 if stopping_iter == -1 else stopping_iter
        self.iteration = 1

        self.nodes = [i for i in range(self.N)]

        self.best_solution = None
        self.best_fitness = float("Inf")
        self.fitness_list = []

    def initial_solution(self):
        """
        Greedy algorithm to get an initial solution (closest-neighbour).
        """
        cur_node = random.choice(self.nodes)  # start from a random node
        solution = [cur_node]

        free_nodes = set(self.nodes)
        free_nodes.remove(cur_node)
        while free_nodes:
            next_node = min(free_nodes, key=lambda x: self.dist(cur_node, x))  # nearest neighbour
            free_nodes.remove(next_node)
            solution.append(next_node)
            cur_node = next_node

        cur_fit = self.fitness(solution)
        if cur_fit < self.best_fitness:  # If best found so far, update best fitness
            self.best_fitness = cur_fit
            self.best_solution = solution
        self.fitness_list.append(cur_fit)
        return solution, cur_fit

    def dist(self, node_0, node_1):
        """
        Euclidean distance between two nodes.
        """
        coord_0, coord_1 = self.coords[node_0], self.coords[node_1]
        return math.sqrt((coord_0[0] - coord_1[0]) ** 2 + (coord_0[1] - coord_1[1]) ** 2)

    def fitness(self, solution):
        """
        Total distance of the current solution path.
        """
        cur_fit = 0
        for i in range(self.N):
            cur_fit += self.dist(solution[i % self.N], solution[(i + 1) % self.N])
        return cur_fit

    def p_accept(self, candidate_fitness):
        """
        Probability of accepting if the candidate is worse than current.
        Depends on the current temperature and difference between candidate and current.
        """
        return math.exp(-abs(candidate_fitness - self.cur_fitness) / self.T)

    
    def diff_better(self, candidate, a,b):
        N = self.N
        
        x = (a+N-1) % self.N
        xx= (x + 1) % self.N
        y = (b+N-1) % self.N
        yy= (y + 1) % self.N
        
        q = self.dist(self.cur_solution[x],  self.cur_solution[xx])
        w = self.dist(self.cur_solution[y],  self.cur_solution[yy])

        e = self.dist(candidate[x], candidate[xx])
        r = self.dist(candidate[y], candidate[yy])

        #print(self.cur_solution)
        #print(x,xx, y,yy)
        #print(self.cur_solution[x],self.cur_solution[xx], self.cur_solution[y],self.cur_solution[yy])
        #print(candidate[x],candidate[xx], candidate[y],candidate[yy])
        
        diff = e+r - (q+w)

        #other_diff = self.fitness(candidate) - self.fitness(self.best_solution)
        #print(diff, "vs", other_diff)

        if q+w > e+r:
            return (True,  diff)
        return     (False, diff)

    
    def accept(self, candidate, a,b):
        """
        Accept with probability 1 if candidate is better than current.
        Accept with probabilty p_accept(..) if candidate is worse.
        """
        #if candidate_length < self.cur_fitness:
       
        diff_good, delta = self.diff_better(candidate, a,b)
        candidate_length = delta + self.cur_fitness
        #candidate_length = self.fitness(candidate)

        if candidate_length < self.cur_fitness:
            self.cur_fitness, self.cur_solution = candidate_length, candidate
            if candidate_length < self.best_fitness:
                self.best_fitness, self.best_solution = candidate_length, candidate
        else:
            if random.random() < self.p_accept( candidate_length ):
                self.cur_fitness, self.cur_solution = candidate_length, candidate

    def anneal(self):
        """
        Execute simulated annealing algorithm.
        """
        # Initialize with the greedy solution.
        self.cur_solution, self.cur_fitness = self.initial_solution()
        
        self.iteration = 1

        print("Starting annealing.")
        while self.T >= self.stopping_temperature and self.iteration < self.stopping_iter:
            candidate = list(self.cur_solution)
            l = random.randint(2, self.N - 1)
            i = random.randint(0, self.N - l)
            #print(candidate)
            candidate[  i : (i + l)  ] = reversed(candidate[  i : (i + l)  ])
            #print(candidate, i,l)
            self.accept(candidate, i, i+l)
            self.T *= self.alpha
            self.iteration += 1
            self.fitness_list.append(self.cur_fitness)

        print("T: ", self.T, "self.stopping_temperature", self.stopping_temperature)
        print("Best fitness obtained: ", self.best_fitness)
        improvement = 100 * (self.fitness_list[0] - self.best_fitness) / (self.fitness_list[0])
        print(f"Improvement over greedy heuristic: {improvement : .2f}%")

    def batch_anneal(self, times=10):
        """
        Execute simulated annealing algorithm `times` times, with random initial solutions.
        """
        for i in range(1, times + 1):
            print(f"Iteration {i}/{times} -------------------------------")
            self.T = self.T_save
            self.iteration = 1
            self.cur_solution, self.cur_fitness = self.initial_solution()
            self.anneal()

    def visualize_routes(self):
        """
        Visualize the TSP route with matplotlib.
        """
        visualize_tsp.plotTSP([self.best_solution], self.coords)

    def plot_learning(self):
        """
        Plot the fitness through iterations.
        """
        plt.plot([i for i in range(len(self.fitness_list))], self.fitness_list)
        plt.ylabel("Fitness")
        plt.xlabel("Iteration")
        plt.show()
