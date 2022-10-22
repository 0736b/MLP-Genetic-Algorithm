from dataclasses import dataclass

@dataclass
class FoldRes:
    train_folds_res: list
    test_folds_res: list
    
    def __init__(self):
        self.train_folds_res = []
        self.test_folds_res = []
        
@dataclass
class GARes:
    max_gen: int
    gens: list
    train_log_avg: list
    train_best_fit: list
    
    def __init__(self):
        self.max_gen = 0
        self.gens = []
        self.train_log_avg = []
        self.train_best_fit = []