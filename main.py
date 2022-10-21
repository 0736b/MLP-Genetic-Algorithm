from ga.ga import GA
from utils.datareader import get_dataset_norm
import sys, os
import time

# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__

def main():
    dataset = get_dataset_norm('dataset/wdbc.data')
    st = time.time()
    ga = GA(50, dataset, 50, [30, 8, 1])
    blockPrint()
    pop,best = ga.run()
    enablePrint()
    mse, acc = best.run_show(dataset)
    print('mse on best model', mse)
    print('acc on best model', acc)
    et = time.time()
    elapsed_time = et - st
    print('Training time:', elapsed_time, 'seconds')

if __name__ == '__main__':
    main()