from ga.ga import GA
from utils.datareader import get_dataset, cross_valid
import time

def main():
    dataset = get_dataset('dataset/wdbc.data', norm=True)
    train_folds, test_folds = cross_valid(dataset)
    folds = 10
    for i in range(folds):
        print('Fold:', (i+1), 'Training')
        st = time.time()
        ga = GA(50, train_folds[i], 20, [30, 8, 1])
        pop, best = ga.run()
        et = time.time()
        elapsed_time = et - st
        print('Fold:', (i+1), 'Training time:', round(elapsed_time, 2), 'seconds')
        print('Fold:', (i+1), 'Testing')
        mse, acc = best.run_show(test_folds[i])
        print('Fold:', (i+1), 'MSE on best individual', round(mse,3), 'Accuracy on best individual', round(acc,3))

if __name__ == '__main__':
    main()