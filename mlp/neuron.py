import numpy as np

def step(x):
    return np.where(x>0, 1,0)

class Neuron:
    
    def __init__(self, prev_layer_neurons: int):
        self.inp = 0.0
        self.output = 0.0
        if prev_layer_neurons != 0:
            self.weights = np.array(np.random.uniform(-1.0,1.0,size=prev_layer_neurons)).reshape((prev_layer_neurons, 1))
        else:
            self.weights = []
        self.bias = 1.0
    
    def set_input(self, _input: int):
        self.inp = _input
    
    def calc_output(self, prev_layer_output: list):
        if not len(prev_layer_output):
            self.output = self.inp
        else:
            self.output = np.array(prev_layer_output)
            self.output = np.dot(self.output, self.weights)
            self.output = step(self.output + self.bias)[0]
        return self.output
    
    def get_output(self):
        return self.output
    
    def get_weights(self):
        return self.weights
    
    def update_weights(self, new_weight, pos):
        self.weights[pos] = new_weight
    
    def clear(self):
        self.inp = 0.0
        self.output = 0.0
        self.bias = 1.0


# if __name__ == '__main__':
#     n = Neuron(3, False)
#     n.set_input(1.0)
#     n.get_output([1.0, 2.0, 3.0])
#     n.get_output([]) 