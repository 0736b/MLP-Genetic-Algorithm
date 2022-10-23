import numpy as np

def step(x):
    return np.where(x>0, 1,0)

class Neuron:
    """Neuron
    """
    
    def __init__(self, prev_layer_neurons: int):
        """Neuron initialize

        Args:
            prev_layer_neurons (int): number of previous layer neurons
        """
        self.inp = 0.0
        self.output = 0.0
        self.prev_layer_neurons = prev_layer_neurons
        if prev_layer_neurons != 0:
            self.weights = np.array(np.random.uniform(-1.0,1.0,size=prev_layer_neurons)).reshape((prev_layer_neurons, 1))
        else:
            self.weights = []
        self.bias = 0.0
    
    def set_input(self, _input: int):
        """set input of neuron

        Args:
            _input (int): input data
        """
        self.inp = _input
    
    def calc_output(self, prev_layer_output: list):
        """calculate output of this neuron

        Args:
            prev_layer_output (list): the result output from previous neurons

        Returns:
            float: output of this neuron
        """
        if not len(prev_layer_output):
            self.output = self.inp
        else:
            self.output = np.array(prev_layer_output)
            self.output = np.dot(self.output, self.weights)
            self.output = step(self.output + self.bias)[0]
        return self.output
    
    def get_output(self):
        """get output value

        Returns:
            float: get output value
        """
        return self.output
    
    def get_weights(self):
        """get weights of this neuron

        Returns:
            list: weights of this neuron
        """
        t_weights = self.weights
        t_weights = t_weights.reshape((1, self.prev_layer_neurons))
        return t_weights[0]
    
    def set_weights(self, new_weights):
        """set new weights on this neuron

        Args:
            new_weights (list): new weights
        """
        self.weights = new_weights.reshape((self.prev_layer_neurons, 1))
    
    def update_weights(self, new_weight, pos):
        """update weight at position

        Args:
            new_weight (float): weight
            pos (int): position of weight
        """
        self.weights[pos] = new_weight