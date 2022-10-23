from ga.ga import GA
from utils.datareader import get_dataset, cross_valid
import time
import pickle

def main():
    dataset = get_dataset('dataset/wdbc.data', norm=True)
    train_folds, test_folds = cross_valid(dataset)
    folds = 10
    for i in range(folds):
        print('Fold:', (i+1), 'Training')
        st = time.time()
        ga = GA(20, train_folds[i], 100, [30, 4, 1])
        best, log_mse_avg, log_mse_best = ga.run()
        et = time.time()
        elapsed_time = et - st
        print('Fold:', (i+1), 'Training time:', round(elapsed_time, 2), 'seconds')
        print('Fold:', (i+1), 'Testing')
        mse, acc = best.run_show(test_folds[i])
        print('Fold:', (i+1), 'MSE on best individual', round(mse,3), 'Accuracy on best individual', round(acc,3))
        # saving model, training log
        # path_model = 'models/30-4-1/best_fold_' + str(i+1) + '.data'
        # path_mse_avg = 'models/30-4-1/ga/mse_avg_fold_' + str(i+1) + '.data'
        # path_mse_best = 'models/30-4-1/ga/mse_best_fold_' + str(i+1) + '.data'
        # with open(path_model, 'wb') as fm:
        #     pickle.dump(best, fm)
        # with open(path_mse_avg, 'wb') as favg:
        #     pickle.dump(log_mse_avg, favg)
        # with open(path_mse_best, 'wb') as fbest:
        #     pickle.dump(log_mse_best, fbest)
        # fm.close()
        # favg.close()
        # fbest.close() 
        
if __name__ == '__main__':
    main()
    