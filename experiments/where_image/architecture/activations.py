from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *

floatX = theano.config.floatX

##################################################################################################################################
class ScaledTanh(Layer):
    
    #################################################################
    def __init__(self, name, in_layer):
        super(ScaledTanh, self).__init__(
                                        name,
                                        children=[in_layer],
                                        dependents=[in_layer.name]
                                    )
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ in_size ] = dependent_sizes
        
        return in_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model ] = dependent_models
        
        return 1.7159*T.tanh(in_model*2/3)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class Softmax(Layer):
    
    #################################################################
    def __init__(self, name, in_layer):
        super(Softmax, self).__init__(
                                        name,
                                        children=[in_layer],
                                        dependents=[in_layer.name]
                                    )
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ in_size ] = dependent_sizes
        
        return in_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model ] = dependent_models
        
        return T.nnet.softmax(in_model)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)