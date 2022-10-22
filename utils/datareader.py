import csv
from sklearn import preprocessing
import numpy as np
import random

def get_dataset(path, norm=False):
    dataset = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data = {}
            inputs = row.copy()
            inputs.pop(0)
            inputs.pop(0)
            inputs = [float(f) for f in inputs]
            inputs_norm = np.array(inputs)
            inputs_norm = preprocessing.normalize([inputs_norm])[0]
            outputs = []
            if norm:
                data['INPUT'] = inputs_norm
            elif not norm:
                data['INPUT'] = inputs
            if row[1] == 'M':
                outputs.append(1)
            else:
                outputs.append(0)
            data['OUTPUT'] = outputs
            dataset.append(data)
    return dataset

def shuffle_data(dataset):
    shuffled = dataset.copy()
    random.shuffle(shuffled)
    return shuffled

def split_groups(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

def cross_valid(dataset):
    shuffled_dataset = shuffle_data(dataset)
    dataset_size = len(shuffled_dataset)
    group_size = int(dataset_size / 10)
    train_dataset_folds = []
    test_dataset_folds = []
    splitted = list(split_groups(shuffled_dataset, group_size))
    if len(splitted) > 10:
        del splitted[10]
    for i in range(len(splitted)):
        sum_fold_train = []
        for j in range(len(splitted)):
            if j != i:
                sum_fold_train += splitted[j].copy()
        fold_test = splitted[i].copy()
        train_dataset_folds.append(sum_fold_train)
        test_dataset_folds.append(fold_test)
    return train_dataset_folds, test_dataset_folds