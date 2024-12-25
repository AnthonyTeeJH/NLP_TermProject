import random

class GAChromosome:
    def __init__(self, dim, lb, ub) -> None:
        self.gene    = [random.uniform(lb, ub) for _ in range(dim)]
        self.fitness = float('inf')
        self.gap = float('inf')
        self.dim = dim
        self.lb = lb
        self.ub = ub