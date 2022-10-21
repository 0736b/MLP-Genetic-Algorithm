import csv
        
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