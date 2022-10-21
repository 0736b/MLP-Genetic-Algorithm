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
        self.children = []
        self.total_fitness = 0
        self.all_fitness = []
        self.prob_pop = []
        self.expect_pop = []
        self.q = []
        self.selected = []
        self.intermediate_pop = []
        self.mating_pool = []
        
    def init_population(self):
        self.population = []
        for i in range(self.pop_size):
            mlp = MLP(self.layers_and_nodes)
            self.population.append(mlp)
        print('now population', self.population)
            
    def calc_fitness(self):
        self.total_fitness = 0
        self.all_fitness = []
        for mlp in self.population:
            fitness = mlp.run(self.dataset)
            self.all_fitness.append(fitness)
            self.total_fitness += fitness
        print('@calc_fitness',self.all_fitness, 'len:', len(self.all_fitness))
        self.avg_fitness = np.average(self.all_fitness)
            
    def selection(self):
        # roulette wheel
        self.prob_pop = []
        self.expect_pop = []
        self.q = []
        self.selected = []
        for fitness in self.all_fitness:
            self.prob_pop.append(fitness/self.total_fitness)
            self.expect_pop.append(fitness/self.avg_fitness)
        print('@selection prob_pop', self.prob_pop, 'len:', len(self.prob_pop))
        for i in range(0,self.pop_size):
            qi = 0
            for k in range(0,i+1):
                qi += self.prob_pop[k]    
            self.q.append(qi)
        print('@selection q', self.q, 'len:', len(self.q))
        for i in range(0, self.pop_size):
            print('@selection i', i)
            rand_num = np.random.uniform(0,1)
            print('@selection rand_num', rand_num)
            if rand_num >= 0 and rand_num <= self.q[0]:
                print('-selected', 0)
                self.selected.append(self.population[0])
            for j in range(1, self.pop_size):
                if rand_num >= self.q[j-1] and rand_num <= self.q[j]:
                    print('-selected', j)
                    self.selected.append(self.population[j])
        print('@selection', self.selected, 'len:', len(self.selected))
    
    def crossover(self, t: int, mate_prob: float):
        # t-point crossver
        self.intermediate_pop = self.selected.copy()
        print('ประชากรกลาง', self.intermediate_pop)
        self.mating_pool = []
        self.children = []
        parents_idx = set([])
        for pop in self.intermediate_pop:
            q = np.random.uniform(0,1)
            if q < mate_prob:
                parents_idx.add(self.intermediate_pop.index(pop))
                self.mating_pool.append(pop)
        if len(self.mating_pool) % 2 != 0:
            rand_num = np.random.uniform(0,1)
            select_idx = np.random.randint(0,len(self.mating_pool))
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
            temp_dad = dad.copy()
            for k in ks:
                dad[k] = mom[k]
                mom[k] = temp_dad[k]
            self.children.append(dad.copy())
            self.children.append(mom.copy())

        t_intermediate_pop = []
        for pop in self.intermediate_pop:
            idx = self.intermediate_pop.index(pop)
            if not (idx in parents_idx):
                t_intermediate_pop.append(pop)
        for pop in self.intermediate_pop:
            print(pop)
            self.children.append(pop.get_chromosome())
        
    def mutation(self, mutate_prob: float):
        # strong mutation
        print('@mutate',self.children)
        for child in self.children:
            for i in range(len(child)):
                q = np.random.uniform(0,1)
                if q < mutate_prob:
                    child[i] = np.random.uniform(-1,1)
    
    def next_gen(self, chromosome: list):
        next_gen = []
        for c in chromosome:
            p = MLP(self.layers_and_nodes)
            p.set_new_weights(c)
            next_gen.append(p)
        self.population = next_gen.copy()
        print('new population',self.population)
        self.current_gen += 1
            
    def run(self):
        self.init_population()
        # while(self.current_gen < self.max_gen):
        self.calc_fitness()
        self.selection()
        self.crossover(3, 0.5)
        self.mutation(0.2)
        self.next_gen(self.children)
        print('curr_gen', self.current_gen)
        return self.population