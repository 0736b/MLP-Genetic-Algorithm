import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils.datareader import get_dataset, cross_valid

def plt_err_trained(model, gen):
    """plot graph between average mse and fittest mse every generations

    Args:
        model (str): model mlp
        gen (int): max generation
    """
    plt.figure(figsize = (20, 10))
    idx_gen = [int(i+1) for i in range(gen)]
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
    
def plt_cfm(color, mode, model):
    """plot confusion matrix of training and validation

    Args:
        color (str): confusion matrix color
        mode (str): training or validation
        model (str): mlp model
    """
    class_output = ['0', '1']   
    params = {
    'axes.titlesize': 16,
    'axes.labelsize': 12,
    'axes.titleweight':'bold',
    'figure.titlesize': 'large'
    }
    mode_text = ''
    if mode == 'train':
        mode_text = 'Training'
        file_mode = 'train'
    elif mode == 'valid':
        mode_text = 'Validation'
        file_mode = 'valid'
    plt.rcParams.update(params)
    plt.figure(figsize = (20,10))
    for i in range(10):
        on_fold = str(i+1)
        path_cfm = 'models/' + model + '/cfm/' + file_mode + '_fold_' + on_fold + '.data'
        with open(path_cfm, 'rb') as f_cfm:
            cfm = pickle.load(f_cfm)
        plt_cfm = cfm.get()
        ax = plt.subplot(2, 5, i+1)
        sns.heatmap(plt_cfm, annot=True, yticklabels=class_output, xticklabels=class_output, cmap=color, fmt='g')
        plt.xlabel('Predicted', fontweight='bold')
        plt.ylabel('Actual', fontweight='bold')
        acc = str(round(cfm.get_accuracy(), 4))
        plt.suptitle('MLP ' + model + ' train with GA' + '\nConfusion Matrix (' + mode_text + ')', fontweight='bold', fontsize=24)
        plt.title('Fold ' + on_fold + ' Accuracy: ' + acc, fontweight='bold')
    plt.subplots_adjust(left=0.06,bottom=0.14,right=0.97,top=0.788,wspace=0.29,hspace=0.51)
    plt.show()
    
def plt_all_mse(max_gen):
    """plot all model mse converge on fold that has highest accuracy

    Args:
        max_gen (int): max generation
    """
    gen = max_gen
    idx_gen = [int(i+1) for i in range(gen)]
    bf41 = '6'
    bf81 = '5'
    bf841 = '4'
    path_4_1_avg = 'models/' + '30-4-1' + '/ga/mse_avg_fold_' + bf41 + '.data'
    path_4_1_best = 'models/' + '30-4-1' + '/ga/mse_best_fold_' + bf41 + '.data'
    path_8_1_avg = 'models/' + '30-8-1' + '/ga/mse_avg_fold_' + bf81 + '.data'
    path_8_1_best = 'models/' + '30-8-1' + '/ga/mse_best_fold_' + bf81 + '.data'
    path_8_4_1_avg = 'models/' + '30-8-4-1' + '/ga/mse_avg_fold_' + bf841 + '.data'
    path_8_4_1_best = 'models/' + '30-8-4-1' + '/ga/mse_best_fold_' + bf841 + '.data'
    with open(path_4_1_avg, 'rb') as f_4_1_avg:
        log_mse_avg_4_1 = pickle.load(f_4_1_avg)
    with open(path_4_1_best, 'rb') as f_4_1_best:
        log_mse_best_4_1 = pickle.load(f_4_1_best)
    with open(path_8_1_avg, 'rb') as f_8_1_avg:
        log_mse_avg_8_1 = pickle.load(f_8_1_avg)
    with open(path_8_1_best, 'rb') as f_8_1_best:
        log_mse_best_8_1 = pickle.load(f_8_1_best)
    with open(path_8_4_1_avg, 'rb') as f_8_4_1_avg:
        log_mse_avg_8_4_1 = pickle.load(f_8_4_1_avg)
    with open(path_8_4_1_best, 'rb') as f_8_4_1_best:
        log_mse_best_8_4_1 = pickle.load(f_8_4_1_best)
    # min_81 = np.amin(log_mse_avg_8_1)
    # max_81 = np.amax(log_mse_avg_8_1)
    # min_81_p = [float(min_81) for i in range(gen)]
    # max_81_p = [float(max_81) for i in range(gen)]
    mse_train_avg41 = pd.DataFrame(log_mse_avg_4_1, index=idx_gen, columns=['30-4-1 Average MSE of All Population'])
    mse_train_best41 = pd.DataFrame(log_mse_best_4_1, index=idx_gen, columns=['30-4-1 MSE of Fittest Individual'])
    mse_train_avg41.index.name = 'Generations'
    mse_train_best41.index.name = 'Generations'
    mse_train_avg81 = pd.DataFrame(log_mse_avg_8_1, index=idx_gen, columns=['30-8-1 Average MSE of All Population'])
    mse_train_best81 = pd.DataFrame(log_mse_best_8_1, index=idx_gen, columns=['30-8-1 MSE of Fittest Individual'])
    mse_train_avg81.index.name = 'Generations'
    mse_train_best81.index.name = 'Generations'
    mse_train_avg841 = pd.DataFrame(log_mse_avg_8_4_1, index=idx_gen, columns=['30-8-4-1 Average MSE of All Population'])
    mse_train_best841 = pd.DataFrame(log_mse_best_8_4_1, index=idx_gen, columns=['30-8-4-1 MSE of Fittest Individual'])
    mse_train_avg841.index.name = 'Generations'
    mse_train_best841.index.name = 'Generations'
    # min_81_data = pd.DataFrame(min_81_p, index=idx_gen, columns=[''])
    # max_81_data = pd.DataFrame(max_81_p, index=idx_gen, columns=[''])
    sns.lineplot(data=mse_train_avg41, linewidth='2.5', palette=['red']).lines[0].set_linestyle("dotted")
    sns.lineplot(data=mse_train_avg81, linewidth='2.5', palette=['blue'])
    sns.lineplot(data=mse_train_avg841, linewidth='2.5', palette=['green']).lines[2].set_linestyle("dotted")
    sns.lineplot(data=mse_train_best41, linewidth='2.5', palette=['red'])
    sns.lineplot(data=mse_train_best81, linewidth='2.5', palette=['blue'])
    sns.lineplot(data=mse_train_best841, linewidth='2.5', palette=['green']).lines[4].set_linestyle("dotted")
    # sns.lineplot(data=min_81_data, palette=['blue'])
    # sns.lineplot(data=max_81_data, palette=['blue'])
    plt.ylabel('Mean Square Error (MSE)', fontsize=18)
    plt.xlabel('Generations', fontsize=18)
    plt.legend(loc='lower right', bbox_to_anchor=(1, 0.15))
    plt.title('30-4-1, 30-8-1, 30-8-4-1 Models\nMSE Converge', fontweight='bold', fontsize=24)
    plt.show()
    
    
if __name__ == '__main__':
    max_gen = 100
    model = '30-8-4-1'
    plt_err_trained(model, max_gen)
    plt_cfm('Blues', 'train', model)
    plt_cfm('YlOrBr', 'valid', model)
    # plt_all_mse(max_gen)