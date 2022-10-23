from ga.ga import GA
from utils.datareader import get_dataset, cross_valid
from utils.confusionmatrix import ConfusionMatrix
import time
import pickle

def main():
    """setups and running
    """
    dataset = get_dataset('dataset/wdbc.data', norm=True)
    train_folds, test_folds = cross_valid(dataset)
    folds = 10
    max_gen = 100
    population = 20
    for i in range(folds):
        model = '30-8-4-1'
        cfm_train = ConfusionMatrix([0,1])
        cfm_valid = ConfusionMatrix([0,1])
        print('Fold:', (i+1), 'Training')
        st = time.time()
        ga = GA(population, train_folds[i], max_gen, [30, 8, 4, 1])
        best, log_mse_avg, log_mse_best = ga.run()
        et = time.time()
        elapsed_time = et - st
        mse_train, acc_train = best.run_show(train_folds[i], cfm_train)
        print('Fold:', (i+1), 'Training time:', round(elapsed_time, 2), 'seconds')
        print('Fold:', (i+1), 'Training Confusion Matrix')
        cfm_train.calc_column()
        cfm_train.print()
        print('Accuracy:', cfm_train.get_accuracy())
        print('Fold:', (i+1), 'Testing')
        mse_valid, acc_valid = best.run_show(test_folds[i], cfm_valid)
        print('Fold:', (i+1), 'MSE on best individual', round(mse_valid,3), 'Accuracy on best individual', round(acc_valid,3))
        print('Fold:', (i+1), 'Validation Confusion Matrix')
        cfm_valid.calc_column()
        cfm_valid.print()
        print('Accuracy:', cfm_valid.get_accuracy())
        # saving model, training log
        path_model = 'models/'+ model + '/best_fold_' + str(i+1) + '.data'
        path_mse_avg = 'models/' + model + '/ga/mse_avg_fold_' + str(i+1) + '.data'
        path_mse_best = 'models/' + model + '/ga/mse_best_fold_' + str(i+1) + '.data'
        path_cfm_train = 'models/' + model + '/cfm/train_fold_' + str(i+1) + '.data'
        path_cfm_valid = 'models/' + model + '/cfm/valid_fold_' + str(i+1) + '.data'
        with open(path_model, 'wb') as fm:
            pickle.dump(best, fm)
        with open(path_mse_avg, 'wb') as favg:
            pickle.dump(log_mse_avg, favg)
        with open(path_mse_best, 'wb') as fbest:
            pickle.dump(log_mse_best, fbest)
        with open(path_cfm_train, 'wb') as fcfm_train:
            pickle.dump(cfm_train, fcfm_train)
        with open(path_cfm_valid, 'wb') as fcfm_valid:
            pickle.dump(cfm_valid, fcfm_valid)
        fm.close()
        favg.close()
        fbest.close()
        fcfm_train.close()
        fcfm_valid.close()
        
if __name__ == '__main__':
    main()
    