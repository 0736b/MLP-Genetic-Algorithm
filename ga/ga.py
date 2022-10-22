from copy import copy
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
        # print('now population', self.population)
    
    def calc_fitness(self):
        self.total_fitness = 0
        self.all_fitness = []
        before_max_fit = -9999999
        best_model = 0
        for mlp in self.population:
            fitness = mlp.run(self.dataset)
            fitness = 1 - fitness
            if fitness >= before_max_fit:
                before_max_fit = fitness
                best_model = mlp
            self.all_fitness.append(fitness)
            self.total_fitness += fitness
        # print('@calc_fitness',self.all_fitness, 'len:', len(self.all_fitness))
        self.avg_fitness = np.average(self.all_fitness)
        return copy(best_model)
    
    def selection(self):
        # roulette wheel
        self.prob_pop = []
        self.expect_pop = []
        self.q = []
        self.selected = []
        # fitness normalize
        for fitness in self.all_fitness:
            self.prob_pop.append(fitness/self.total_fitness)
            self.expect_pop.append(fitness/self.avg_fitness)
        # print('@selection prob_pop', self.prob_pop, 'len:', len(self.prob_pop))
        for i in range(0,self.pop_size):
            qi = 0
            for k in range(0,i+1):
                qi += self.prob_pop[k]    
            self.q.append(qi)
        # print('@selection q', self.q, 'len:', len(self.q))
        for i in range(0, self.pop_size):
            # print('@selection i', i)
            rand_num = np.random.uniform(0,1)
            # print('@selection rand_num', rand_num)
            if rand_num >= 0 and rand_num <= self.q[0]:
                # print('-selected', 0)
                self.selected.append(self.population[0])
            for j in range(1, self.pop_size):
                if rand_num >= self.q[j-1] and rand_num <= self.q[j]:
                    # print('-selected', j)
                    self.selected.append(self.population[j])
        # print('@selection', self.selected, 'len:', len(self.selected))
    
    def crossover(self, t: int, mate_prob: float):
        # t-point crossver
        self.intermediate_pop = self.selected
        # print('@crossover ประชากรกลาง', self.intermediate_pop, 'len:', len(self.intermediate_pop))
        self.mating_pool = []
        self.children = []
        parents_idx = set([])
        for pop in self.intermediate_pop:
            q = np.random.uniform(0,1)
            if q < mate_prob:
                parents_idx.add(self.intermediate_pop.index(pop))
                self.mating_pool.append(pop)
        if len(self.mating_pool) % 2 != 0:
            # rand_num = np.random.uniform(0,1)
            #random between add or remove
            # if rand_num > 0.5:
            #     select_idx = np.random.randint(0,len(self.mating_pool))
            #     self.mating_pool.pop(select_idx)
            # else:
            select_idx = np.random.randint(0,len(self.intermediate_pop))
            self.mating_pool.append(self.intermediate_pop[select_idx])
            parents_idx.add(select_idx)
        # print('@crossover คณะพ่อแม่', self.mating_pool, 'len:', len(self.mating_pool))
        for i in range(0, len(self.mating_pool), 2):
            dad = self.mating_pool[i].get_chromosome()
            mom = self.mating_pool[i+1].get_chromosome()
            ks = [np.random.randint(0, len(dad) - 1) for i in range(t)]
            # print('@crossover สุ่มค่า k', ks, 'len:', len(ks))
            # swap
            temp_dad = dad.copy()
            # temp_dad2 = dad.copy()
            # temp_mom = mom.copy()
            for k in ks:
                dad[k] = mom[k]
                mom[k] = temp_dad[k]
            self.children.append(dad)
            self.children.append(mom)
        # print('@crossover ลูกที่ได้จากการจับคู่พ่อแม่', self.children, 'len:', len(self.children))
        t_intermediate_pop = []
        # print('@crossover p1', self.intermediate_pop, 'len:', len(self.intermediate_pop))
        for pop in self.intermediate_pop:
            idx = self.intermediate_pop.index(pop)
            if not (idx in parents_idx):
                t_intermediate_pop.append(pop)
        self.intermediate_pop = t_intermediate_pop.copy()
        # print('@crossover p1 หลังเอาพ่อแม่ออก', self.intermediate_pop, 'len:', len(self.intermediate_pop))
        for pop in self.intermediate_pop:
            self.children.append(pop.get_chromosome())
        # print('@crossover ลูกเติมด้วย p1 ที่เหลือ', self.children, 'len:', len(self.children))
        
    def mutation(self, mutate_prob: float):
        # strong mutation
        # print('@mutate ก่อน',self.children, 'len', len(self.children))
        for child in self.children:
            for i in range(len(child)):
                q = np.random.uniform(0,1)
                if q < mutate_prob:
                    child[i] = np.random.uniform(-1,1)
        # print('@mutate หลัง',self.children, 'len', len(self.children))
    
    def next_gen(self, chromosome: list, elitism):
        next_gen = []
        for c in chromosome:
            p = MLP(self.layers_and_nodes)
            p.set_new_weights(c)
            next_gen.append(p)
        self.population = next_gen
        self.population.append(elitism)
        self.pop_size = len(self.population)
        self.current_gen += 1
        
    def find_best(self):
        model = self.calc_fitness()
        return copy(model)
    
    def run(self):
        self.init_population()
        while(self.current_gen < self.max_gen):
            elitism = self.calc_fitness()
            self.selection()
            self.crossover(5, 0.8)
            self.mutation(0.1)
            self.next_gen(self.children, elitism)
            print('curr_gen', self.current_gen, 'mse:', elitism.run(self.dataset))
        best = self.find_best()
        return self.population, best 