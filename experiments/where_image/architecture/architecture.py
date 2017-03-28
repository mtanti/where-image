from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import json
import numpy as np

floatX = theano.config.floatX

##################################################################################################################################
class Architecture(object):
    '''
    Networks are defined by using a list of tree structures. Each tree consists of layer objects connected together with the leaves being the inputs and the root being the output. Several trees can be used to have several outputs and the layers can be shared using Ref layers.
    
    Mods are there to take the same network and modify its outputs in order to have a variety of different uses such as calculating a cost function or getting a single probability from a softmax. Mods are specified using a dictionary
    In the case of testing mods:
        { mod_name: (input_names, output_names, mod_function) }
        where mod_name is the name that is used to refer to the mod, input_names is list of input layer names that will be used by the architecture of the mod, output_names is a list of output layer names from the network that are fed to the mod function, and mod_function is the mod function to use.
    In the case of training mods:
        { mod_name: (input_names, output_names, mod_function, optimizer) }
        where optimizer is the optimizer object to use when training and mod_function would be a cost function.
    '''
    
    #################################################################
    def __init__(self, network_list, testing_mods, training_mods=dict()):
        self.training_mods = dict()
        self.testing_mods = dict()
        self.layers = dict()
        self.params = list()
        
        def enum_layers(layer):
            yield (layer.name, layer)
            for child in layer.children:
                for pair in enum_layers(child):
                    yield pair
        root_layer_names = list()
        input_layer_names = set()
        layers_used = set()
        for network in network_list:
            root_layer_names.append(network.name)
            for (name, layer) in enum_layers(network):
                if name in self.layers:
                    raise ValueError('Duplicate name "{}" found in network.'.format(layer.name))
                if layer in layers_used:
                    raise ValueError('Layer "{}" used multiple times in network. Use a "Ref" layer to refer to existing layers.'.format(layer.name))
                self.layers[name] = layer
                if len(layer.dependents) == 0:
                    input_layer_names.add(name)
                layers_used.add(layer)
        del layers_used
        
        errors_found = []
        for (mod_type_name, mods) in [('testing mod', testing_mods), ('training mod', training_mods)]:
            for (mod_name, mod_tuple) in mods.items():
                (input_order, output_order) = mod_tuple[:2]
                for input_name in input_order:
                    if input_name not in input_layer_names:
                        errors_found.append((input_name, 'input order', mod_type_name, mod_name))
                for output_name in output_order:
                    if output_name not in self.layers:
                        errors_found.append((output_name, 'output order', mod_type_name, mod_name))
        if len(errors_found) > 0:
            raise ValueError('The following mentioned layer names were not found in the network_list:' + ''.join('\n\t"{0}" in the {1} part of the {2} "{3}"'.format(*error) for error in errors_found))
            
        layer_names_used = set()
        def flatten_layers(layer_names):
            for layer_name in layer_names:
                if layer_name not in layer_names_used:
                    layer_names_used.add(layer_name)
                    dependents = self.layers[layer_name].dependents
                    for descendent_name in flatten_layers(dependents):
                        yield descendent_name
                yield layer_name
        
        layer_sizes = dict()
        testing_models = dict()
        training_models = dict()
        for layer_name in flatten_layers(root_layer_names):
            if layer_name in testing_models or layer_name in training_models:
                continue
                
            layer = self.layers[layer_name]
            
            layer_sizes[layer_name] = layer.compile_params([ layer_sizes[name] for name in layer.dependents ])
            
            self.params.extend(layer.params)
            
            if len(testing_mods) > 0:
                testing_models[layer_name] = layer.get_testing_model([ testing_models[name] for name in layer.dependents ])
            
            if len(training_mods) > 0:
                training_models[layer_name] = layer.get_training_model([ training_models[name] for name in layer.dependents ])
            
        for (mod_name, (input_order, output_order, mod_function)) in testing_mods.items():
            outputs = mod_function(*[ testing_models[name] for name in output_order ])
            self.testing_mods[mod_name] = {
                                                'outputs': outputs,
                                                'inputs': [ testing_models[name] for name in input_order ],
                                                'input_names': input_order,
                                                'function': None
                                            }
        
        for (mod_name, (input_order, output_order, mod_function, optimizer)) in training_mods.items():
            outputs = mod_function(*[ training_models[name] for name in output_order ])
            if len(outputs) != 1:
                raise ValueError('Training mod "{}" gives more than one output. Training mods can only give one output.'.format(mod_name))
            self.training_mods[mod_name] = {
                                                'outputs': outputs,
                                                'inputs': [ training_models[name] for name in input_order ],
                                                'input_names': input_order,
                                                'gradient': T.grad(outputs[0], self.params),
                                                'optimizer': optimizer,
                                                'function': None
                                            }

    #################################################################
    def compile(self):
        for mod in self.testing_mods.values():
            mod['function'] = theano.function(mod['inputs'], mod['outputs'])
        for mod in self.training_mods.values():
            mod['optimizer'].compile(self.params, mod['gradient'])
            mod['function'] = theano.function(mod['inputs'], mod['outputs'], updates=mod['optimizer'].next_update_list())
        self.init()
    
    #################################################################
    def testf(self, mod_name):
        if self.testing_mods[mod_name]['function'] is None:
            raise ValueError('Testing functions have not been compiled yet.')
        return self.testing_mods[mod_name]['function']
    
    #################################################################
    def trainf(self, mod_name):
        if self.training_mods[mod_name]['function'] is None:
            raise ValueError('Training functions have not been compiled yet.')
        return self.training_mods[mod_name]['function']
    
    #################################################################
    def set_params(self, params):
        common_layer_names = self.layers.keys() & params.keys()
        for layer_name in common_layer_names:
            self.layers[layer_name].set_params([ np.array(p, dtype=floatX) for p in params[layer_name] ])
        return common_layer_names
    
    #################################################################
    def get_params(self):
        return { layer_name : [ p.get_value().tolist() for p in layer.params ] for (layer_name, layer) in self.layers.items() if layer.params != [] }
    
    #################################################################
    def init(self):
        for layer in self.layers.values():
            layer.init_params()
        for mod in self.training_mods.values():
            mod['optimizer'].init()
    
    #################################################################
    def get_num_params(self):
        return sum(p.get_value().size for (layer_name, layer) in self.layers.items() for p in layer.params)
    