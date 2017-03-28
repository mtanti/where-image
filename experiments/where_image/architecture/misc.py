from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *

floatX = theano.config.floatX

##################################################################################################################################
class Ref(Layer):
    
    #################################################################
    def __init__(self, name, referred_name):
        super(Ref, self).__init__(
                                        name,
                                        dependents=[referred_name]
                                    )
    
    #################################################################
    def compile_params(self, dependent_sizes):
        [ in_size ] = dependent_sizes
        
        return in_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model ] = dependent_models
        
        return in_model
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
##################################################################################################################################
class LastItem(Layer):
    
    #################################################################
    def __init__(self, name, seq):
        super(LastItem, self).__init__(
                                        name,
                                        children=[seq],
                                        dependents=[seq.name]
                                    )
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ in_size ] = dependent_sizes
        
        return in_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model ] = dependent_models
        
        return in_model.dimshuffle((1,0,2))[-1]
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
##################################################################################################################################
class Dropout(Layer):
    
    srng = T.shared_randomstreams.RandomStreams()
    
    #################################################################
    def __init__(self, name, dropout_rate, in_layer):
        super(Dropout, self).__init__(
                                        name,
                                        children=[in_layer],
                                        dependents=[in_layer.name]
                                    )
        
        self.dropout_rate = dropout_rate
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ in_size ] = dependent_sizes
        
        return in_size
    
    #################################################################
    def get_training_model(self, dependent_models):
        [ in_model ] = dependent_models
        
        #see _dropout_from_layer in https://github.com/mdenil/dropout/blob/master/mlp.py
        dropmask_generator = Dropout.srng.binomial(n=1, p=1-self.dropout_rate, size=in_model.shape)
        return in_model * T.cast(dropmask_generator/(1-self.dropout_rate), floatX)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        [ in_model ] = dependent_models
        
        return in_model