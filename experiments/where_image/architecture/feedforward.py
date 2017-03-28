from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *
from architecture.initializations import *

floatX = theano.config.floatX

##################################################################################################################################
class Dense(Layer):
    
    #################################################################
    def __init__(self, name, output_size, in_layer):
        super(Dense, self).__init__(
                                        name,
                                        children=[in_layer],
                                        dependents=[in_layer.name]
                                    )
        
        self.output_size = output_size
        self.W = None
        self.b = None
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ in_size ] = dependent_sizes
        
        self.W = create_param((in_size, self.output_size), self.name+'_W')
        self.b = create_param((self.output_size,), self.name+'_b')
        self.params.extend([ self.W, self.b ])
        
        return self.output_size
    
    #################################################################
    def init_params(self):
        apply_init(xavier_normal, self.W)
        apply_init(zeros, self.b)
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model ] = dependent_models
        
        return T.dot(in_model, self.W) + self.b
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class EmbeddedSeqs(Layer):
    
    #################################################################
    def __init__(self, name, token_vector_size, indexseqs_layer):
        super(EmbeddedSeqs, self).__init__(
                                            name,
                                            children=[indexseqs_layer],
                                            dependents=[indexseqs_layer.name]
                                        )
        
        self.token_vector_size = token_vector_size
        self.token_vectors = None
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ vocab_size ] = dependent_sizes
        
        self.token_vectors = create_param((vocab_size, self.token_vector_size), self.name+'_W')
        self.params.extend([ self.token_vectors ])
        
        return self.token_vector_size
    
    #################################################################
    def init_params(self):
        apply_init(xavier_normal, self.token_vectors)
    
    #################################################################
    def _get_model(self, dependent_models):
        [ indexseqs ] = dependent_models
        
        return self.token_vectors[indexseqs]
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)