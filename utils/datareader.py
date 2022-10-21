import csv
from sklearn import preprocessing
import numpy as np
import random

def get_dataset(path):
    dataset = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data = {}
            inputs = row.copy()
            inputs.pop(0)
            inputs.pop(0)
            inputs = [float(f) for f in inputs]
            outputs = []
            data['INPUT'] = inputs
            if row[1] == 'M':
                outputs.append(1.0)
            else:
                outputs.append(0.0)
            data['OUTPUT'] = outputs
            dataset.append(data)
    return dataset

def get_dataset_norm(path):
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
            data['INPUT'] = inputs_norm
            if row[1] == 'M':
                outputs.append(1)
            else:
                outputs.append(0)
            data['OUTPUT'] = outputs
            dataset.append(data)
    return dataset

def get_xor(path):
    dataset = []
    with open(path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            data = {}
            inputs = row.copy()
            inputs.pop(0)
            inputs.pop(0)
            inputs = [float(f) for f in inputs]
            outputs = []
            data['INPUT'] = inputs
            if row[1] == '1':
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