from mlp.mlp import MLP
import numpy as np

class GA:
    
    def __init__(self, pop_size: int, dataset, max_gen: int, layers_and_nodes: list):
        self.pop_size = pop_size
        self.population = []
        self.dataset = dataset
        self.current_gen = 1
        self.max_gen = max_gen
        self.layers_and_nodes = layers_and_nodes
        
    def init_population(self):
        self.population = []
        for i in range(self.pop_size):
            mlp = MLP(self.layers_and_nodes)
            self.population.append(mlp)
            
    def calc_fitness(self):
        self.total_fitness = 0
        self.all_fitness = []
        for mlp in self.population:
            fitness = mlp.run(self.dataset)
            self.all_fitness.append(fitness)
            self.total_fitness += fitness
        self.avg_fitness = np.average(self.all_fitness)
            
    def selection(self):
        # roulette wheel
        self.prob_pop = []
        self.expect_pop = []
        self.q = []
        self.selected = []
        for fitness in self.all_fitness:
            self.prop_pop.append(fitness/self.total_fitness)
            self.expect_pop.append(fitness/self.avg_fitness)
        for i in range(0,self.pop_size):
            qi = 0
            for k in range(0,i):
                qi += self.prob_pop[k]    
            self.q.append(qi)
        for i in range(0, self.pop_size):
            rand_num = np.random.uniform(0,1)
            if rand_num >= 0 and rand_num <= self.q[0]:
                self.selected.append(self.population[i])
            if rand_num >= self.q[i-1] and rand_num <= self.q[i]:
                self.selected.append(self.population[i-1])
    
    def crossover(self, t: int, mate_prob: float):
        # t-point crossver
        self.intermediate_pop = self.selected
        self.mating_pool = []
        self.children = []
        parents_idx = {}
        for pop in self.intermediate_pop:
            q = np.random.uniform(0,1)
            if q < mate_prob:
                parents_idx.add(self.intermediate_pop.index(pop))
                self.mating_pool.append(pop)
        if len(self.mating_pool) % 2 != 0:
            rand_num = np.random.uniform(0,1)
            select_idx = np.random.randint(0,self.pop_size - 1)
            # random between add or remove
            if rand_num > 0.5:
                self.mating_pool.pop(select_idx)
                parents_idx.remove(select_idx)
            else:
                self.mating_pool.append(self.intermediate_pop[select_idx])
                parents_idx.add(select_idx)
        for i in range(0, len(self.mating_pool), 2):
            dad = self.mating_pool[i].get_chromosome()
            mom = self.mating_pool[i+1].get_chromosome()
            ks = [np.random.randint(0, len(dad) - 1) for i in range(t)]
            # swap
            temp_dad = dad
            for k in ks:
                dad[k] = mom[k]
                mom[k] = temp_dad[k]
            parents = [dad, mom]
            for p in parents:
                # child = MLP(self.layers_and_nodes)
                # child.set_new_weights(p)
                self.children.append(p)
        for idx in parents_idx:
            self.intermediate_pop.pop(idx)
        for pop in self.intermediate_pop:
            self.children.append(pop.get_chromosome())
        # self.new_population = self.children
        # self.new_population += self.intermediate_pop
        
    def mutation(self, mutate_prob: float):
        # strong mutation
        for child in self.children:
            for i in range(len(child)):
                q = np.random.uniform(0,1)
                if q < mutate_prob:
                    child.update_weights(np.random.uniform(-1,1),i)
    
    def next_gen(self, chromosome: list):
        next_gen = []
        for c in chromosome:
            p = MLP(self.layers_and_nodes)
            p.set_new_weights(c)
            next_gen.append(p)
        self.population = next_gen
        self.current_gen += 1
            
    def run(self):
        self.init_population()
        while(self.current_gen < self.max_gen):
            self.calc_fitness()
            self.selection()
            self.crossover(3, 0.5)
            self.mutation(0.2)
            self.next_gen(self.children)
        return self.population