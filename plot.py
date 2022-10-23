import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from utils.datareader import get_dataset, cross_valid

def plt_err_trained(model):
    plt.figure(figsize = (20, 10))
    idx_gen = [int(i+1) for i in range(100)]
    for i in range(10):
        on_fold = str(i+1)
        path_avg = 'models/' + model + '/ga/mse_avg_fold_' + on_fold + '.data'
        path_best = 'models/' + model + '/ga/mse_best_fold_' + on_fold + '.data'
        with open(path_avg, 'rb') as f_avg:
            log_mse_avg = pickle.load(f_avg)
        with open(path_best, 'rb') as f_best:
            log_mse_best = pickle.load(f_best)
        plt.subplot(2,5,i+1)
        mse_train_avg = pd.DataFrame(log_mse_avg, index=idx_gen, columns=[''])
        mse_train_best = pd.DataFrame(log_mse_best, index=idx_gen, columns=[''])
        mse_train_avg.index.name = 'Generations'
        mse_train_best.index.name = 'Generations'
        sns.lineplot(data=mse_train_avg, palette=['black'])
        sns.lineplot(data=mse_train_best, palette=['red'])
        plt.ylabel('Mean Square Error (MSE)')
        plt.suptitle('MLP ' + model + ' train with GA' + '\nMSE Converge', fontweight='bold', fontsize=24)
        plt.title('Fold ' + on_fold + ', MSE on best individual: ' + str(round(log_mse_best[len(log_mse_best) - 1],3)), fontweight='bold')
        f_avg.close()
        f_best.close()
    plt.subplots_adjust(left=0.04,bottom=0.117,right=0.97,top=0.817,wspace=0.29,hspace=0.51)
    plt.show()
    
def plt_valid(model):
    dataset = get_dataset('dataset/wdbc.data', norm=True)
    train_folds, test_folds = cross_valid(dataset)
    folds = 10
    
if __name__ == '__main__':
    plt_err_trained('30-4-1')
        
