from chromosome import GAChromosome as Chromosome
from SCH import SCH_func
import random
import time
import pickle
import math

POP_SIZE = 100
XOVER_RATE = 0.9
GENERATION = 5000
EVAL_TERMINATION = 30000

class SGA:
    def __init__(self, problem, task_dim, lower_bound, upper_bound):

        #### Variable declaration ####
        self.lb = lower_bound
        self.ub = upper_bound

        self.dim = task_dim #int
        self.pop_size = POP_SIZE #int
        self.xover_rate  = XOVER_RATE #float
        self.mutate_rate = 1/task_dim #float
        self.generations = GENERATION #int

        self.problem = problem

        self.pop = []
        self.best_so_far = None
        self.gen = 0
        self.evaluation = 0
        self.historical_data = []

        self.start_time = time.time()

        self.pop = [Chromosome(dim=self.dim, lb=self.lb, ub=self.ub) for i in range(self.pop_size)]
        self.best_so_far = self.pop[0]

        #### Evaluate the initial population ####
        for _, chrm in enumerate(self.pop):
            # print(f"Evaluating {(i/len(offspring))*100: 3.1f}%", end="\r")
            chrm.fitness = self.evaluate(chrm.gene)
            self.evaluation += 1
            if chrm.fitness < self.best_so_far.fitness: # Minimization
                self.best_so_far = chrm
            info = {'BestFitness': self.best_so_far.fitness, 'Generation':self.gen, 'Time': time.time()-self.start_time}
            self.historical_data.append(info)

    def evolve(self):
        self.gen += 1
        offspring = []

        #### Reproduction ####
        while len(offspring) < self.pop_size:
            
            #### Parent selection ####
            p1 = self.parent_selection()
            p2 = self.parent_selection()
            #### Global crossover ####
            c1, c2 = self.uniform_xover(p1,p2)
            
            #### Mutation ####
            c1 = self.mutation(c1)
            c2 = self.mutation(c2)

            offspring += [c1, c2]
        
        #### Evaluate offspring ####
        for _, chrm in enumerate(offspring):
            # print(f"Evaluating {(i/len(offspring))*100: 3.1f}%", end="\r")
            chrm.fitness = self.evaluate(chrm.gene)
            self.evaluation += 1
            if chrm.fitness < self.best_so_far.fitness: # Minimization
                self.best_so_far = chrm
            info = {'BestFitness': self.best_so_far.fitness, 'Generation':self.gen, 'Time': time.time()-self.start_time}
            self.historical_data.append(info)

        #### Survival selection ####
        self.pop = self.survival_selection(offspring)
    
    def parent_selection(self):
        # Calculate the sum of fitness values
        total_fitness = sum(1/chrm.fitness for chrm in self.pop)
        # Calculate the probability distribution
        probabilities = [(1/chrm.fitness)/total_fitness for chrm in self.pop]
        # Spin the roulette wheel to select one individual
        selected_individual = random.choices(self.pop, probabilities)[0]
        return selected_individual
    
    def uniform_xover(self, p1, p2):
        child1, child2 = Chromosome(self.dim,self.lb,self.ub), Chromosome(self.dim,self.lb,self.ub)
        
        if random.random() <= self.xover_rate:
            for i in range(len(p1.gene)):
                if random.random() <= 0.5:
                    child1.gene[i] = p1.gene[i]
                    child2.gene[i] = p2.gene[i]
                else:
                    child1.gene[i] = p2.gene[i]
                    child2.gene[i] = p1.gene[i]
        else:
            child1.gene = p1.gene[:]
            child2.gene = p2.gene[:]
        
        return child1, child2
        
    def mutation(self, chrm):
        for i in range(self.dim):
            if random.random() <= self.mutate_rate:
                chrm.gene[i] = random.uniform(self.lb, self.ub)
        return chrm

    def evaluate(self, gene) -> float:
        obj_value = self.problem.test(gene[:self.dim])
        # obj_value = SCH_func(gene[:self.dim])
        return obj_value

    def survival_selection(self, offspring_pool):
        # mu + lambda
        n_survivor = len(self.pop)
        sorted_pool = sorted(offspring_pool, key=lambda chrm:chrm.fitness, reverse=False)
        return sorted_pool[:n_survivor]
    
    def optimize(self, run, result_dir, record_data=False, display=False) -> float:
        if display:
            print(f"Generation {0: 4d}, eval num {(self.evaluation): 4d}, best fitness = {self.best_so_far.fitness:4.4f}")
        ### Evolve the population ###
        for gen in range(self.generations):
            self.evolve()
            if display:
                print(f"Generation {(gen+1): 4d}, eval num {(self.evaluation): 4d}, best fitness = {self.best_so_far.fitness:4.4f}")
            if self.evaluation > EVAL_TERMINATION:
                break
        if record_data:
            with open(result_dir + 'run_' + str(run) + '.pickle', 'wb') as f:
                pickle.dump(self.historical_data, f)
        return self.best_so_far.fitness, self.evaluation, self.best_so_far.fitness