from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *

floatX = theano.config.floatX

##################################################################################################################################
class IndexSeqsInput(Layer):
    
    #################################################################
    def __init__(self, name, vocab_size):
        super(IndexSeqsInput, self).__init__(name)
        
        self.vocab_size = vocab_size
        
    #################################################################
    def compile_params(self, dependent_sizes):
        assert dependent_sizes == []
        
        return self.vocab_size
        
    #################################################################
    def _get_model(self):
        return T.matrix(name=self.name, dtype='int16')
    
    #################################################################
    def get_training_model(self, dependent_models):
        assert dependent_models == []
        
        return self._get_model()
        
    #################################################################
    def get_testing_model(self, dependent_models):
        assert dependent_models == []
        
        return self._get_model()
        

##################################################################################################################################
class IntsInput(Layer):
    
    #################################################################
    def __init__(self, name):
        super(IntsInput, self).__init__(name)
        
    #################################################################
    def compile_params(self, dependent_sizes):
        if dependent_sizes != []:
            raise ValueError('Input layer cannot depend on anything.')
        
        return 1
        
    #################################################################
    def _get_model(self):
        return T.vector(name=self.name, dtype='int64')
    
    #################################################################
    def get_training_model(self, dependent_models):
        if dependent_models != []:
            raise ValueError('Input layer cannot depend on anything.')
        
        return self._get_model()
        
    #################################################################
    def get_testing_model(self, dependent_models):
        if dependent_models != []:
            raise ValueError('Input layer cannot depend on anything.')
        
        return self._get_model()
        
        
##################################################################################################################################
class VectorsInput(Layer):
    
    #################################################################
    def __init__(self, name, vector_size):
        super(VectorsInput, self).__init__(name)
        
        self.vector_size = vector_size
        
    #################################################################
    def compile_params(self, dependent_sizes):
        if dependent_sizes != []:
            raise ValueError('Input layer cannot depend on anything.')
        
        return self.vector_size
        
    #################################################################
    def _get_model(self):
        return T.matrix(name=self.name, dtype=floatX)
    
    #################################################################
    def get_training_model(self, dependent_models):
        if dependent_models != []:
            raise ValueError('Input layer cannot depend on anything.')
        
        return self._get_model()
        
    #################################################################
    def get_testing_model(self, dependent_models):
        if dependent_models != []:
            raise ValueError('Input layer cannot depend on anything.')
        
        return self._get_model()