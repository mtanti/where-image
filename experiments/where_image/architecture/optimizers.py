from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *

floatX = theano.config.floatX

##################################################################################################################################
class Optimizer(object):

    #################################################################
    def __init__(self):
        pass
    
    #################################################################
    def compile(self, params, grads):
        raise NotImplementedError()

    #################################################################
    def init(self):
        raise NotImplementedError()

    #################################################################
    def next_update_list(self):
        raise NotImplementedError()
    
##################################################################################################################################
class Adam(Optimizer):

    #################################################################
    def __init__(self, gradient_clipping_magnitude=None, learning_rate=0.001, epsilon=1e-8, beta1=0.9, beta2=0.999):
        super(Adam, self).__init__()
        
        self.gradient_clipping_magnitude = gradient_clipping_magnitude
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2
        
        self.params = None
        self.grads = None
        self.ms = None
        self.vs = None
        self.t = None
    
    #################################################################
    def compile(self, params, grads):
        self.params = params
        if self.gradient_clipping_magnitude is None:
            self.grads = grads
        else:
            self.grads = [ T.clip(g, -self.gradient_clipping_magnitude, self.gradient_clipping_magnitude) for g in grads ]
        self.ms = [ theano.shared(np.zeros_like(p.get_value(), dtype=floatX)) for p in self.params ]
        self.vs = [ theano.shared(np.zeros_like(p.get_value(), dtype=floatX)) for p in self.params ]
        self.t = theano.shared(np.array(0, dtype='int64'))
        self.init()

    #################################################################
    def init(self):
        for m in self.ms:
            m.set_value(np.zeros_like(m.get_value(), dtype=floatX))
        for v in self.vs:
            v.set_value(np.zeros_like(v.get_value(), dtype=floatX))
        self.t.set_value(np.array(1, dtype='int64'))

    #################################################################
    def next_update_list(self):
        if self.params is None:
            raise ValueError('Optimizer has not been compiled yet.')
            
        new_ms = [ self.beta1*m + (1 - self.beta1)*g        for (m,g) in zip(self.ms, self.grads) ]
        new_vs = [ self.beta2*v + (1 - self.beta2)*T.sqr(g) for (v,g) in zip(self.vs, self.grads) ]
        
        ms_hat = [ m/T.cast(1 - T.pow(self.beta1, self.t), floatX) for m in new_ms ]
        vs_hat = [ v/T.cast(1 - T.pow(self.beta2, self.t), floatX) for v in new_vs ]
        
        return (
                [ (m, new_m) for (m,new_m) in zip(self.ms, new_ms) ] +
                [ (v, new_v) for (v,new_v) in zip(self.vs, new_vs) ] +
                [ (self.t, self.t + 1) ] +
                [ (p, p - self.learning_rate*m_hat/(T.sqrt(v_hat) + self.epsilon)) for (p,m_hat,v_hat) in zip(self.params, ms_hat, vs_hat) ]
            )
