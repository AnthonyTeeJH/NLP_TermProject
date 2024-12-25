import random
import os
from parent_select_operators import random_parent_selection, tournament_parent_selection, fps_parent_selection
from xover_operators import whole_arithmetic_xover, single_arithmetic_xover, simple_xover, uniform_xover, n_points_xover, two_points_xover, one_point_xover
from mutate_operators import uniform_mutate, nonuniform_mutate
from survival_select_operators import random_survival_selection, tournament_survival_selection, fps_survival_selection, elitism_survival_selection, generational_survival_selection
from chromosome import GAChromosome as Chromosome
import time
import pickle
import math

#### GA Parameters ####
GA_POPULATION_SIZE = 100
GA_GENERATION      = 5000
GA_K_TOURNAMENT    = 4
GA_CROSSOVER_RATE  = 0.9
# GA_MUTATION_RATE   = 1/CHRM_DIM
EVAL_TERMINATION = 30000

class Solver:
    def __init__(self, algo_components, problem, task_dim, lower_bound, upper_bound):

        # print(algo_components) solver_sga_100 = [2, 6, 1, 2, 5, 2, 1, 1, 1] 2,2,6,
        parent_select_op ={
            0: random_parent_selection,
            1: tournament_parent_selection,
            2: fps_parent_selection
        }
        xover_op = {
            0: 0, 
            1: whole_arithmetic_xover,
            2: single_arithmetic_xover,
            3: simple_xover,
            4: uniform_xover
            # 5: two_points_xover,
            # 6: one_point_xover
        }
        mutate_op = {
            0: 0,
            1: uniform_mutate,
            2: nonuniform_mutate,
        }
        survival_select_op ={
            0: random_survival_selection,
            1: elitism_survival_selection,
            2: generational_survival_selection
        }
        tournament_size ={
            1:2,
            2:4,
            3:8,
            4:16,
            5:32,
            6:64
        }
        # xover_arithmetic_alpha ={
        #     1: 0.2,
        #     2: 0.5,
        #     3: 0.8
        # }
        # mutate_nonuniform_sigma ={
        #     1: 1,
        #     2: 1.5,
        #     3: 3
        # }
        crossover_rate ={
            1: 0.1,
            2: 0.3,
            3: 0.6,
            4: 0.9
        }
        mutation_rate = { #x/length
            1: 0.1,
            2: 0.5,
            3: 1,
            4: 3,
            5: 6,
            6: 9
        }
        population_size ={ #mu
            1: 40,
            2: 80,
            3: 160,
            4: 320,
            5: 640
        }
        offspring_size ={ #倍數
            1: 0.5,
            2: 1,
            3: 2,
            4: 3,
            5: 4,
            6: 5,
            7: 6,
            8: 7
        }
        # print(algo_components)
        #### Variable declaration ####
        
        # Operators
        self.parent_select_op = parent_select_op.get(algo_components[0])
        self.xover_op = xover_op.get(algo_components[1])
        self.mutate_op = mutate_op.get(algo_components[2])
        self.survival_select_op = survival_select_op.get(algo_components[3])
        # Parameters
        self.tournament_size = tournament_size.get(algo_components[4])
        self.xover_rate = crossover_rate.get(algo_components[5])
        self.mutate_rate = mutation_rate.get(algo_components[6])/task_dim
        self.pop_size = population_size.get(algo_components[7])
        self.offspring_size = offspring_size.get(algo_components[8])
        
        self.xover_alpha = 0.8 #xover_arithmetic_alpha.get(algo_components[6])
        self.mutate_sigma = 1.0 #mutate_nonuniform_sigma.get(algo_components[7])
    
        self.dim = task_dim
        self.lb = lower_bound
        self.ub = upper_bound
        self.pop = []
        # self.xover_rate  = GA_CROSSOVER_RATE
        # self.mutate_rate = 1.0/task_dim
        self.best_so_far = None
        self.evaluation = 0
        self.problem = problem
        self.problem_opt = 0
        self.gen = 0
        self.historical_data_eval_num = []
        self.start_time = time.time()

        #### Initialize variables ####
        self.pop = [Chromosome(dim=self.dim, lb=self.lb, ub=self.ub) for i in range(self.pop_size)]
        self.best_so_far = self.pop[0]

        #### Evaluate the initial population ####
        for i, chrm in enumerate(self.pop):
            # print(f"Evaluating {(i/self.pop_size)*100: 3.1f}%", end="\r")
            chrm.fitness = evaluate(chrm.gene, self.problem)
            chrm.gap = chrm.fitness/100
            self.evaluation += 1

            if chrm.fitness < self.best_so_far.fitness:
                self.best_so_far = chrm
        info = {'Problem':self.problem,'Eval':self.evaluation, 'BestFitness': self.best_so_far.fitness, 'BestGap': self.best_so_far.gap, 'Mode':'Gen', 'Generation':self.gen, 'Time': time.time()-self.start_time}
        self.historical_data_eval_num.append(info)

    def evolve(self):
        self.gen += 1
        offspring = []

        # for i, chrm in enumerate(self.pop):
        #     # print(f"Evaluating {(i/self.pop_size)*100: 3.1f}%", end="\r")
        #     print("Chromosome: ", chrm.gene)
        #     print("Fitness: ", chrm.fitness)

        #### Reproduction ####
        while len(offspring) < self.offspring_size*self.pop_size:

            #### Parent selection ####
            op_params = [self.pop, self.tournament_size]
            if self.parent_select_op == None:
                print(self.tournament_size)
                print(self.parent_select_op)
            p1 = self.parent_select_op(*op_params)
            p2 = self.parent_select_op(*op_params)

            #### Crossover ####
            if self.xover_op != 0:
                op_params = [p1, p2, self.xover_rate, self.xover_alpha]
                c1, c2 = self.xover_op(*op_params)
                #### Mutation ####
                if self.mutate_op != 0:
                    op_params = [c1, self.mutate_rate, self.mutate_sigma, self.lb, self.ub]
                    c1 = self.mutate_op(*op_params)
                    op_params = [c2, self.mutate_rate, self.mutate_sigma, self.lb, self.ub]
                    c2 = self.mutate_op(*op_params)
            else:
                #### Mutation ####
                if self.mutate_op != 0:
                    op_params = [p1, self.mutate_rate, self.mutate_sigma, self.lb, self.ub]
                    c1 = self.mutate_op(*op_params)
                    op_params = [p2, self.mutate_rate, self.mutate_sigma, self.lb, self.ub]
                    c2 = self.mutate_op(*op_params)
                else: # No crossover and mutation -> random create offspring
                    c1 = Chromosome(dim=self.dim, lb=self.lb, ub=self.ub)
                    c2 = Chromosome(dim=self.dim, lb=self.lb, ub=self.ub)
            
            offspring += [c1, c2]
        
        #### Evaluate offspring ####
        for i, chrm in enumerate(offspring):
            # print(f"Evaluating {(i/self.pop_size)*100: 3.1f}%", end="\r")
            chrm.fitness = evaluate(chrm.gene,self.problem)
            chrm.gap = chrm.fitness/100
            self.evaluation += 1

            if chrm.fitness < self.best_so_far.fitness:
                self.best_so_far = chrm
        info = {'Problem':self.problem, 'Eval':self.evaluation, 'BestFitness': self.best_so_far.fitness, 'BestGap': self.best_so_far.gap, 'Mode':'Gen', 'Generation':self.gen, 'Time': time.time()-self.start_time}
        self.historical_data_eval_num.append(info)

        #### Survival selection ####
        op_params = [self.pop, offspring]
        self.pop = self.survival_select_op(*op_params)

    def optimize(self, run, result_dir, record_data=False, round=0, display=False) -> float:
        if display:
            print()
            print(f"Generation {0: 4d}, Eval num {(self.evaluation): 4d}, best fitness = {self.best_so_far.fitness:4.4f}, best gap = {self.best_so_far.gap:4.4f}")
        ### Evolve the population ###
        for gen in range(GA_GENERATION):
            self.evolve()
            if display:
                print(f"Generation {(gen+1): 4d}, Eval num {(self.evaluation): 4d}, best fitness = {self.best_so_far.fitness:4.4f}, best gap = {self.best_so_far.gap:4.4f}")
            if self.evaluation > EVAL_TERMINATION:
                #print("Termination criteria met! #Evaluation = ", self.evaluation)
                break
        if record_data:
            with open(result_dir + 'round_' + str(round) + '_run_' + str(run) + '.pickle', 'wb') as f:
                pickle.dump(self.historical_data_eval_num, f)
        
        return self.best_so_far.fitness, self.evaluation
    
def evaluate(gene:list, problem) -> float:
    obj_value = problem.test(gene)
    # obj_value = SCH_func(gene)
    return obj_value

def test(gene):
    parent_select_op ={
        0: random_parent_selection,
        1: tournament_parent_selection,
        2: fps_parent_selection
    }
    xover_op = {
        1: whole_arithmetic_xover,
        2: single_arithmetic_xover,
        3: simple_xover
    }
    mutate_op = {
        1: uniform_mutate,
        2: nonuniform_mutate,
    }
    survival_select_op ={
        0: random_survival_selection,
        1: tournament_survival_selection,
        2: fps_survival_selection,
        3: elitism_survival_selection,
        4: generational_survival_selection
    }
    parent_select_op ={
        0: random_parent_selection,
        1: tournament_parent_selection,
        2: fps_parent_selection
    }
    xover_op = {
        1: whole_arithmetic_xover,
        2: single_arithmetic_xover,
        3: simple_xover
    }
    mutate_op = {
        1: uniform_mutate,
        2: nonuniform_mutate,
    }
    survival_select_op ={
        0: random_survival_selection,
        1: tournament_survival_selection,
        2: fps_survival_selection,
        3: elitism_survival_selection,
        4: generational_survival_selection
    }
    # pop_size, parent_selection, xover, mutate, survival_selection
    if gene[2] == 0:
        xover = 0
    else:
        xover = xover_op.get(gene[2])
    if gene[3] == 0:
        mutate = 0
    else:
        mutate = mutate_op.get(gene[3])
    algo_components = [gene[0], parent_select_op.get(gene[1]), xover, mutate, survival_select_op.get(gene[4])]
    result = Solver(algo_components, 10, -512.0, 511.0)
    return result