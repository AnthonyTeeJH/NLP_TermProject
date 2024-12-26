from chromosome import GAChromosome as Chromosome
import numpy as np
import random
import pickle

class GA:
    def __init__(self, dim, lb, ub, target_vec, gen, pop_size, xover_rate, mutate_rate, k):
        """
        GA class for NLP Term Project (Prompt Recovery).
        
        Parameters:
            dim (int): embedding vector length.
            lb (float): embedding space lower bound value.
            ub (float): embedding space upper bound value.
            target_vec: target embedding vector.
            gen (int): number of generation.
            pop_size (int): population size.
            xover_rate (float [0,1]): crossover rate.
            mutate_rate (float [0,1]): mutation rate.
            k (int): tournament size.
        Target:
            Obtain the embedding vector with the highest similarity to the given target embedding vector(s).
        """
        #### Variable declaration ####
        self.lb = lb
        self.ub = ub

        self.dim = dim
        self.pop_size = pop_size
        self.xover_rate  = xover_rate
        self.mutate_rate = mutate_rate
        self.generation = gen
        self.k = k
        self.target_vec = target_vec

        self.pop = []
        self.best_so_far = None
        self.gen = 0
        self.historical_data = []

        self.pop = [Chromosome(dim=self.dim, lb=self.lb, ub=self.ub) for i in range(self.pop_size)]
        self.best_so_far = self.pop[0]

        #### Evaluate the initial population ####
        for _, chrm in enumerate(self.pop):
            chrm.fitness = self.evaluate(chrm.gene)
            if chrm.fitness > self.best_so_far.fitness:
                self.best_so_far = chrm
            info = {'BestFitness': self.best_so_far.fitness, 'Generation':self.gen}
            self.historical_data.append(info)

    def evolve(self):
        self.gen += 1
        offspring = []

        #### Reproduction ####
        while len(offspring) < self.pop_size:
            
            #### Parent selection ####
            p1 = self.parent_selection(self.k)
            p2 = self.parent_selection(self.k)
            #### Global crossover ####
            c1, c2 = self.uniform_xover(p1,p2)
            
            #### Mutation ####
            c1 = self.mutation(c1)
            c2 = self.mutation(c2)

            offspring += [c1, c2]
        
        #### Evaluate offspring ####
        for _, chrm in enumerate(offspring):
            chrm.fitness = self.evaluate(chrm.gene)
            if chrm.fitness > self.best_so_far.fitness: 
                self.best_so_far = chrm
            info = {'BestFitness': self.best_so_far.fitness, 'Generation':self.gen}
            self.historical_data.append(info)

        #### Survival selection ####
        self.pop = self.survival_selection(offspring)

    def parent_selection(self, k) -> Chromosome:
        candidates = []
        for i in range(k): #k-tournament
            candidates.append(self.pop[random.randint(0, (self.pop_size-1))])
        sorted_candidates = sorted(candidates, key=lambda chrm:chrm.fitness, reverse=True)
        return sorted_candidates[0]
    
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
        if len(self.target_vec) > 1:
            fitness_list = []
            for prompt in self.target_vec:
                fitness_list.append(sharpened_cosine_similarity(gene, self.target_vec[0], 3))
            fitness = np.mean(fitness_list)
        else:
            fitness = sharpened_cosine_similarity(gene, self.target_vec[0], 3)
        return fitness

    def survival_selection(self, offspring_pool):
        # mu + lambda
        n_survivor = len(self.pop)
        sorted_pool = sorted(offspring_pool, key=lambda chrm:chrm.fitness, reverse=True)
        return sorted_pool[:n_survivor]
    
    def optimize(self, run, result_dir, record_data=False, display=False) -> float:
        if display:
            print(f"Generation {0: 4d}, best fitness = {self.best_so_far.fitness:4.4f}")
        ### Evolve the population ###
        for gen in range(self.generation):
            self.evolve()
            if display:
                print(f"Generation {(gen+1): 4d}, best fitness = {self.best_so_far.fitness:4.4f}")
        if record_data:
            with open(result_dir + '/run_' + str(run) + '.pickle', 'wb') as f:
                pickle.dump(self.historical_data, f)
        return self.best_so_far.gene, self.best_so_far.fitness

def sharpened_cosine_similarity(u, v, alpha=3):
    """
    Compute the Sharpened Cosine Similarity between two vectors.
    
    Parameters:
        u (np.ndarray): First vector.
        v (np.ndarray): Second vector.
        alpha (float): Sharpening parameter (default is 3).
        
    Returns:
        float: Sharpened cosine similarity.
    """
    # Ensure the vectors are NumPy arrays
    u = np.array(u)
    v = np.array(v)
    
    # Compute the cosine similarity
    cosine_sim = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    
    # Apply the sharpening exponent
    return cosine_sim**alpha