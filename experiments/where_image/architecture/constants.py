from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *

floatX = theano.config.floatX

##################################################################################################################################
class Zeros(Layer):
    
    #################################################################
    def __init__(self, name, vector_size, rows_like):
        super(Zeros, self).__init__(
                                        name,
                                        children=[rows_like],
                                        dependents=[rows_like.name]
                                    )
        
        self.vector_size = vector_size
    
    #################################################################
    def compile_params(self, dependent_sizes):
        [ rows_like_size ] = dependent_sizes
        
        return self.vector_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ rows_like_model ] = dependent_models
        
        return T.zeros( (rows_like_model.shape[0], self.vector_size), floatX )
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
