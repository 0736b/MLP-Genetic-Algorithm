from copy import copy
from mlp.mlp import MLP
import numpy as np

class GA:
    """Genetic Algorithm
    """
    def __init__(self, pop_size: int, dataset, max_gen: int, layers_and_nodes: list):
        """GA Initialize

        Args:
            pop_size (int): population size
            dataset (list): dataset to train
            max_gen (int): maximum generations
            layers_and_nodes (list): layers and nodes for MLP model 
        """
        self.pop_size = pop_size
        self.population = []
        self.dataset = dataset
        self.current_gen = 1
        self.max_gen = max_gen
        self.log_mse_avg = []
        self.log_mse_best = []
        self.layers_and_nodes = layers_and_nodes

    def init_population(self):
        """initialize population with size of population
        """
        self.population = []
        for i in range(self.pop_size):
            mlp = MLP(self.layers_and_nodes)
            self.population.append(mlp)
    
    def calc_fitness(self):
        """calculate fitness

        Returns:
            MLP: MLP that fittest
        """
        self.total_fitness = 0
        self.all_fitness = []
        before_max_fit = -9999999
        best_model = 0
        for mlp in self.population:
            fitness = mlp.run(self.dataset)
            fitness = 1 - fitness
            if fitness >= before_max_fit:
                before_max_fit = fitness
                self.max_fit = fitness
                self.min_mse = 1 - fitness
                best_model = mlp
            self.all_fitness.append(fitness)
            self.total_fitness += fitness
        self.avg_fitness = np.average(self.all_fitness)
        self.avg_mse = 1 - self.avg_fitness
        return copy(best_model)
    
    def selection(self):
        """doing Selection with roulette wheel method
        """
        # roulette wheel
        self.prob_pop = []
        self.expect_pop = []
        self.q = []
        self.selected = []
        # fitness normalize
        for fitness in self.all_fitness:
            self.prob_pop.append(fitness/self.total_fitness)
            self.expect_pop.append(fitness/self.avg_fitness)
        for i in range(0,self.pop_size):
            qi = 0
            for k in range(0,i+1):
                qi += self.prob_pop[k]    
            self.q.append(qi)
        for i in range(0, self.pop_size):
            rand_num = np.random.uniform(0,1)
            if rand_num >= 0 and rand_num <= self.q[0]:
                self.selected.append(self.population[0])
            for j in range(1, self.pop_size):
                if rand_num >= self.q[j-1] and rand_num <= self.q[j]:
                    self.selected.append(self.population[j])
    
    def crossover(self, t: int, mate_prob: float):
        """doing t-point Crossover

        Args:
            t (int): t-point
            mate_prob (float): mating probability
        """
        # t-point crossver
        self.intermediate_pop = self.selected
        self.mating_pool = []
        self.children = []
        parents_idx = set([])
        for pop in self.intermediate_pop:
            q = np.random.uniform(0,1)
            if q < mate_prob:
                parents_idx.add(self.intermediate_pop.index(pop))
                self.mating_pool.append(pop)
        if len(self.mating_pool) % 2 != 0:
            select_idx = np.random.randint(0,len(self.intermediate_pop))
            self.mating_pool.append(self.intermediate_pop[select_idx])
            parents_idx.add(select_idx)
        for i in range(0, len(self.mating_pool), 2):
            dad = self.mating_pool[i].get_chromosome()
            mom = self.mating_pool[i+1].get_chromosome()
            ks = [np.random.randint(0, len(dad) - 1) for i in range(t)]
            # swap
            temp_dad = dad.copy()
            for k in ks:
                dad[k] = mom[k]
                mom[k] = temp_dad[k]
            self.children.append(dad)
            self.children.append(mom)
        t_intermediate_pop = []
        for pop in self.intermediate_pop:
            idx = self.intermediate_pop.index(pop)
            if not (idx in parents_idx):
                t_intermediate_pop.append(pop)
        self.intermediate_pop = t_intermediate_pop.copy()
        for pop in self.intermediate_pop:
            self.children.append(pop.get_chromosome())
        
    def mutation(self, mutate_prob: float):
        """doing Strong Mutation

        Args:
            mutate_prob (float): mutation probability
        """
        # strong mutation
        for child in self.children:
            for i in range(len(child)):
                q = np.random.uniform(0,1)
                if q < mutate_prob:
                    child[i] = np.random.uniform(-1,1)
    
    def next_gen(self, chromosome: list, elitism):
        """set next generation GA parameter

        Args:
            chromosome (list): children and intermediate population chromosomes
            elitism (MLP): send the fittest to next gen
        """
        next_gen = []
        for c in chromosome:
            p = MLP(self.layers_and_nodes)
            p.set_new_weights(c)
            next_gen.append(p)
        self.population = next_gen
        self.population.append(elitism)
        self.pop_size = len(self.population)
        
    def find_best(self):
        """find the fittest

        Returns:
            MLP: MLP model with fittest individual
        """
        model = self.calc_fitness()
        return copy(model)
    
    def run(self):
        """trainning with genetic algorithm

        Returns:
            MLP: best MLP with fittest after trainning to max generation
        """
        self.init_population()
        while(self.current_gen <= self.max_gen):
            elitism = self.calc_fitness()
            self.selection()
            self.crossover(5, 0.8)
            self.mutation(0.1)
            self.next_gen(self.children, elitism)
            print('Training @Gen', self.current_gen, 'MSE/ACC on best individual:', round(self.min_mse,4), '/',round(self.max_fit * 100, 4), 'AVG MSE of Population:', round((self.avg_mse),4))
            self.log_mse_avg.append(self.avg_mse)
            self.log_mse_best.append(self.min_mse)
            self.current_gen += 1
        best = self.find_best()
        return best, self.log_mse_avg, self.log_mse_best