from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *

floatX = theano.config.floatX

##################################################################################################################################
class MergeAdd(Layer):
    
    #################################################################
    def __init__(self, name, in_layer1, in_layer2):
        super(MergeAdd, self).__init__(
                                        name,
                                        children=[in_layer1, in_layer2],
                                        dependents=[in_layer1.name, in_layer2.name]
                                    )
        
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ input1_size, input2_size ] = dependent_sizes
        
        if input1_size != input2_size:
            raise ValueError('Layers must have the same output size in order to be merged additively.')
        
        return input1_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model1, in_model2 ] = dependent_models
        
        return in_model1 + in_model2
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        

##################################################################################################################################
class MergeMult(Layer):
    
    #################################################################
    def __init__(self, name, in_layer1, in_layer2):
        super(MergeMult, self).__init__(
                                        name,
                                        children=[in_layer1, in_layer2],
                                        dependents=[in_layer1.name, in_layer2.name]
                                    )
        
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ input1_size, input2_size ] = dependent_sizes
        
        if input1_size != input2_size:
            raise ValueError('Layers must have the same output size in order to be merged multiplicatively.')
        
        return input1_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model1, in_model2 ] = dependent_models
        
        return in_model1 * in_model2
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class MergeConcat(Layer):
    
    #################################################################
    def __init__(self, name, in_layer1, in_layer2):
        super(MergeConcat, self).__init__(
                                        name,
                                        children=[in_layer1, in_layer2],
                                        dependents=[in_layer1.name, in_layer2.name]
                                    )
    
    #################################################################
    def compile_params(self, dependent_sizes):
        [ input1_size, input2_size ] = dependent_sizes
        
        return input1_size + input2_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ in_model1, in_model2 ] = dependent_models
        
        return T.concatenate([in_model1, in_model2], axis=1)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        

##################################################################################################################################
class ParInjectSeq(Layer):
    
    #################################################################
    def __init__(self, name, new_items, in_layer):
        super(ParInjectSeq, self).__init__(
                                        name,
                                        children=[new_items, in_layer],
                                        dependents=[new_items.name, in_layer.name]
                                    )
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ new_item_size, in_size ] = dependent_sizes
        
        if new_item_size != in_size:
            raise ValueError('Layers must have the same output size in order to be merged in parallel.')
        
        return in_size
        
    #################################################################
    def _get_model(self, dependent_models):
        [ new_items, in_model ] = dependent_models
        
        vectors_to_join = T.extra_ops.repeat(new_items.dimshuffle(0,'x',1), in_model.shape[1], axis=1)
        return vectors_to_join + in_model
        
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class PreInjectSeq(Layer):
    
    #################################################################
    def __init__(self, name, new_items, in_layer):
        super(PreInjectSeq, self).__init__(
                                        name,
                                        children=[new_items, in_layer],
                                        dependents=[new_items.name, in_layer.name]
                                    )
    
    #################################################################
    def compile_params(self, dependent_sizes):
        [ new_item_size, in_size ] = dependent_sizes
        
        return in_size
    
    #################################################################
    def _get_model(self, dependent_models):
        [ new_items, in_model ] = dependent_models
        
        return T.concatenate([ new_items.dimshuffle(0,'x',1), in_model ], axis=1)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class PreInjectMask(Layer):
    
    #################################################################
    def __init__(self, name, value, mask_layer):
        super(PreInjectMask, self).__init__(
                                        name,
                                        children=[mask_layer],
                                        dependents=[mask_layer.name]
                                    )
        self.value = value
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ mask_size ] = dependent_sizes
        
        return mask_size
        
    #################################################################
    def _get_model(self, dependent_models):
        [ mask ] = dependent_models
        
        return T.concatenate([ self.value*T.ones((mask.shape[0], 1), 'int16'), mask ], axis=1)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class PostInjectSeq(Layer):
    
    #################################################################
    def __init__(self, name, new_items, in_layer):
        super(PostInjectSeq, self).__init__(
                                        name,
                                        children=[new_items, in_layer],
                                        dependents=[new_items.name, in_layer.name]
                                    )
    
    #################################################################
    def compile_params(self, dependent_sizes):
        [ new_item_size, in_size ] = dependent_sizes
        
        return in_size
        
    #################################################################
    def _get_model(self, dependent_models):
        [ new_items, in_model ] = dependent_models
        
        return T.concatenate([ in_model, new_items.dimshuffle(0,'x',1) ], axis=1)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class PostInjectMask(Layer):
    
    #################################################################
    def __init__(self, name, value, mask_layer):
        super(PostInjectMask, self).__init__(
                                        name,
                                        children=[mask_layer],
                                        dependents=[mask_layer.name]
                                    )
        self.value = value
    
    #################################################################
    def compile_params(self, dependent_sizes):
        [ mask_size ] = dependent_sizes
        
        return mask_size
        
    #################################################################
    def _get_model(self, dependent_models):
        [ mask ] = dependent_models
        
        return T.concatenate([ mask, self.value*T.ones((mask.shape[0], 1), 'int16') ], axis=1)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
