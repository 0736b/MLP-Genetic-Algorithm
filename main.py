from ga.ga import GA
from utils.datareader import get_dataset
import time

def main():
    dataset = get_dataset('dataset/wdbc.data', norm=True)
    print(dataset)
    st = time.time()
    ga = GA(50, dataset, 50, [30, 8, 1])
    pop, best = ga.run()
    mse, acc = best.run_show(dataset)
    print('mse on best model', mse)
    print('acc on best model', acc)
    et = time.time()
    elapsed_time = et - st
    print('Training time:', elapsed_time, 'seconds')

if __name__ == '__main__':
    main()