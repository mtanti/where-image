from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

floatX = theano.config.floatX

##################################################################################################################################
class Layer(object):
    
    #################################################################
    def __init__(self, name, children=[], dependents=[]):
        self.name = name
        self.children = children
        self.dependents = dependents
        
        self.params = []
    
    #cannot store parameters or model in constructor as model might be a reference to another node which is unknown before the whole network is constructed
    
    #################################################################
    def compile_params(self, dependent_sizes):
        raise NotImplementedError()
    
    #################################################################
    def share_params(self, source_layer):
        raise ValueError('A unparametered layer cannot share parameters.')
    
    #################################################################
    def set_params(self, params):
        for (param, new_param) in zip(self.params, params):
            param.set_value(new_param)
    
    #################################################################
    def init_params(self):
        pass
    
    #################################################################
    def get_training_model(self, dependent_models):
        raise NotImplementedError()
        
    #################################################################
    def get_testing_model(self, dependent_models):
        raise NotImplementedError()
        

def create_param(shape, name):
    return theano.shared(np.empty(shape, dtype=floatX), name=name)
    