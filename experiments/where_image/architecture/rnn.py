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
class SimpleRnn(Layer):
    
    #################################################################
    def __init__(self, name, state_size, pad_value, init_state, in_mask, in_layer):
        super(SimpleRnn, self).__init__(
                                        name,
                                        children=[init_state, in_mask, in_layer],
                                        dependents=[init_state.name, in_mask.name, in_layer.name]
                                    )
        
        self.state_size = state_size
        self.pad_value = pad_value
        self.W_h = None
        self.b   = None
    
    #################################################################
    def compile_params(self, dependent_sizes):
        [ init_states_size, mask_size, in_size ] = dependent_sizes
        
        self.W_h = create_param((self.state_size, self.state_size), self.name+'_W_h')
        self.b   = create_param((self.state_size,), self.name+'_b')
        self.params.extend([ self.W_h, self.b ])
        
        return self.state_size
    
    #################################################################
    def init_params(self):
        apply_init(orthogonal, self.W_h)
        apply_init(zeros, self.b)

    #################################################################
    def _get_model(self, dependent_models):
        [ init_states, mask, in_model ] = dependent_models
        
        #Shuffle the dimensions of the input vectors so that the first element contains the first word of every sentence, the second element contains the second word of every sentence, etc.
        #Let 's' be sentence, 'w' be word, and 'f' be word feature (element in vector)
        #The dimshuffle converts [[[s1w1f1,s1w1f2],[s1w2f1,s1w2f2]],[[s2w1f1,s2w1f2],[s2w2f1,s2w2f2]] into 
        #                        [[[s1w1f1,s1w1f2],[s2w1f1,s2w1f2]],[[s1w2f1,s1w2f2],[s2w2f1,s2w2f2]]
        outputs = theano.scan(
                            SimpleRnn._step,
                            sequences=[ in_model.dimshuffle((1,0,2)), mask.dimshuffle((1,0,'x')) ],
                            outputs_info=[ init_states ],
                            non_sequences=[ self.W_h, self.b, self.pad_value ],
                            strict=True
                        )[0]
        return outputs[-1]
    
    @staticmethod
    def _step(curr_ins, input_masks, curr_states, W_h, b, pad_value):
        x = curr_ins
        h = curr_states
        h_ = T.nnet.relu(x + T.dot(h, W_h) + b)
        return T.switch(T.eq(input_masks, pad_value), h, h_)
    
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        
##################################################################################################################################
class Lstm(Layer):
    #This is an LSTM as defined by Vinyals for Show and Tell neural image captioning.
    #It works using the following equations:
    #   Let 'x' be the current input, 'h' be the current hidden state, 'c' be the current cell state,
    #       'i' be the input gate, 'f' be the forget gate, 'o' be the output gate, 'h'' be the next hidden state, 'c'' be the next cell state
    #   i  = sig(x.W_xi + h.W_hi + b_i)
    #   f  = sig(x.W_xf + h.W_hf + b_f)
    #   c' = f*c + i*tanh(x.W_xc + h.W_hc + b_c)
    #   o  = sig(x.W_xo + h.W_ho + b_o)
    #   h' = o*c'
    #
    #As an optimisation, each of the following lines is calculated in one matrix multiplication by concatenating the weights column-wise and then extracting the separate answers from the resulting vector using vector slicing
    #   x.W_xi, x.W_xf, x.W_xc, x.W_xo = x.(W_xi ++ W_xf ++ W_xc ++ W_xo) = x.W_x_ifco = x_ifco
    #   h.W_hi, h.W_hf, h.W_hc, h.W_ho = h.(W_hi ++ W_hf ++ W_hc ++ W_ho) = h.W_h_ifco = h_ifco
    #   and b_i, b_f, b_c, b_o are all concatenated into b_ifco so that
    #       x_ifco + h_ifco + b_ifco = (x.W_xi+h.W_hi+b_i) ++ (x.W_xf+h.W_hf+b_f) ++ (x.W_xc+h.W_hc+b_c) ++ (x.W_xo+h.W_ho+b_o) = xhb_ifco
    
    #################################################################
    def __init__(self, name, state_size, pad_value, init_state, in_mask, in_layer):
        super(Lstm, self).__init__(
                                        name,
                                        children=[init_state, in_mask, in_layer],
                                        dependents=[init_state.name, in_mask.name, in_layer.name]
                                    )

        
        self.state_size = state_size
        self.in_size = None
        self.pad_value = pad_value
        self.W_x_ifco = None
        self.W_h_ifco = None
        self.b_ifco = None
        
    #################################################################
    def compile_params(self, dependent_sizes):
        [ init_states_size, mask_size, in_size ] = dependent_sizes
        self.in_size  = in_size
        self.W_x_ifco = create_param((self.in_size, self.state_size*4), self.name+'_W_x_ifco')
        self.W_h_ifco = create_param((self.state_size, self.state_size*4), self.name+'_W_h_ifco')
        self.b_ifco   = create_param((self.state_size*4,), self.name+'_b_ifco')
        self.params.extend([ self.W_x_ifco, self.W_h_ifco, self.b_ifco ])
        
        return self.state_size
    
    #################################################################
    def init_params(self):
        apply_init(lambda shape:np.concatenate([ xavier_normal((shape[0], shape[1]//4)) for _ in range(4) ], axis=1), self.W_x_ifco)
        apply_init(lambda shape:np.concatenate([ orthogonal((shape[0], shape[1]//4)) for _ in range(4) ], axis=1), self.W_h_ifco)
        apply_init(zeros, self.b_ifco)
    
    #################################################################
    def _get_model(self, dependent_models):
        [ init_states_h, mask, in_model ] = dependent_models
        
        init_states_c = T.zeros( (in_model.shape[0], self.state_size), floatX )
        
        #Shuffle the dimensions of the input vectors so that the first element contains the first word of every sentence, the second element contains the second word of every sentence, etc.
        #Let 's' be sentence, 'w' be word, and 'f' be word feature (element in vector)
        #The dimshuffle converts [[[s1w1f1,s1w1f2],[s1w2f1,s1w2f2]],[[s2w1f1,s2w1f2],[s2w2f1,s2w2f2]] into 
        #                        [[[s1w1f1,s1w1f2],[s2w1f1,s2w1f2]],[[s1w2f1,s1w2f2],[s2w2f1,s2w2f2]]
        outputs = theano.scan(
                            Lstm._step,
                            sequences=[ in_model.dimshuffle((1,0,2)), mask.dimshuffle((1,0,'x')) ],
                            outputs_info=[ init_states_h, init_states_c ],
                            non_sequences=[ self.W_x_ifco, self.W_h_ifco, self.b_ifco, self.pad_value, self.state_size ],
                            strict=True
                        )[0][0]
        return outputs[-1]
    
    @staticmethod
    def _step(curr_ins, input_masks, curr_states, curr_cells, W_x_ifco, W_h_ifco, b_ifco, pad_value, state_size):
        x = curr_ins
        h = curr_states
        c = curr_cells
        
        xhb_ifco = T.dot(x, W_x_ifco) + T.dot(h, W_h_ifco) + b_ifco
        xhb_i = xhb_ifco[:, 0*state_size:1*state_size]
        xhb_f = xhb_ifco[:, 1*state_size:2*state_size]
        xhb_c = xhb_ifco[:, 2*state_size:3*state_size]
        xhb_o = xhb_ifco[:, 3*state_size:4*state_size]
        
        i = T.nnet.sigmoid(xhb_i)
        f = T.nnet.sigmoid(xhb_f)
        c_ = f*c + i*T.tanh(xhb_c)
        o = T.nnet.sigmoid(xhb_o)
        h_ = o*c_
        return [
                T.switch(T.eq(input_masks, pad_value), h, h_),
                T.switch(T.eq(input_masks, pad_value), c, c_)
            ]
        
    #################################################################
    def get_training_model(self, dependent_models):
        return self._get_model(dependent_models)
        
    #################################################################
    def get_testing_model(self, dependent_models):
        return self._get_model(dependent_models)
        
        