import numpy as np
from neuron import Neuron

class MLP:
    
    def __init__(self, layers_and_nodes: list):
        self.layers = {}
        self.sse = []
        for i in range(len(layers_and_nodes)):
            t_layers = []
            for j in range(layers_and_nodes[i]):
                if i == 0:
                    t_layers.append(Neuron(0))
                else:
                    t_layers.append(Neuron(layers_and_nodes[i-1]))
            if i == 0:
                self.layers['INPUT_LAYER'] = np.array(t_layers)
            elif i == len(layers_and_nodes) - 1:
                self.layers['OUTPUT_LAYER'] = np.array(t_layers)
            else:
                key = 'HIDDEN_LAYER' + str(i)
                self.layers[key] = np.array(t_layers)
    
    def forward_pass(self, input_data: list):
        if len(input_data) != len(self.layers['INPUT_LAYER']):
            print('input data not match input neurons')
            return
        else:
            prev_layer_output = []
            for i in range(len(input_data)):
                self.layers['INPUT_LAYER'][i].set_input(input_data[i])
                prev_layer_output.append(input_data[i])
            for layer, neurons in self.layers.items():
                if 'HIDDEN_LAYER' in layer or 'OUTPUT_LAYER' in layer:
                    o_from_hidden = []
                    for neuron in neurons:
                        output = neuron.calc_output(prev_layer_output)
                        o_from_hidden.append(output)
                    prev_layer_output = o_from_hidden

    def calc_error(self, desired_output: list):
        error = 0.0
        for i in range(len(self.layers['OUTPUT_LAYER'])):
            error += (self.layers['OUTPUT_LAYER'][i].get_output() - desired_output[i])
        return error
    
    def run(self, dataset):
        sse = []
        for train_data in dataset:
            self.forward_pass(train_data['INPUT'])
            error = self.calc_error(train_data['OUTPUT'])
            sse.append(error)
        return 1/(1+np.average(sse))
    
    def get_chromosome(self):
        nodes = []
        for layer, neurons in self.layers.items():
            if 'HIDDEN_LAYER' in layer or 'OUTPUT_LAYER' in layer:
                for neuron in neurons:
                    nodes.append(neuron)
        return nodes
    
    def set_new_weights(self, new_weights: list):
        idx = 0
        for i in range(len(self.layers_and_nodes)):
            t_layers = []
            for j in range(self.layers_and_nodes[i]):
                if i == 0:
                    t_layers.append(Neuron(0))
                else:
                    new_weights[idx].clear()
                    t_layers.append(new_weights[idx])
                    idx += 1
            if i == 0:
                self.layers['INPUT_LAYER'] = np.array(t_layers)
            elif i == len(self.layers_and_nodes) - 1:
                self.layers['OUTPUT_LAYER'] = np.array(t_layers)
            else:
                key = 'HIDDEN_LAYER' + str(i)
                self.layers[key] = np.array(t_layers)
                    