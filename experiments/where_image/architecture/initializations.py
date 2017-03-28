from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

floatX = theano.config.floatX

#################################################################
def identity(shape):
    return np.identity(shape[0], dtype=floatX)

#################################################################
def xavier_normal(shape):
    (fan_in, fan_out) = shape
    scale = np.sqrt(2 / (fan_in + fan_out))
    values = np.random.normal(0.0, scale, shape)
    return np.asarray(values, dtype=floatX)

#################################################################
def orthogonal(shape):
    rand = np.random.normal(0.0, 1.0, shape)
    (u, _, v) = np.linalg.svd(rand, full_matrices=False)
    if u.shape == shape:
        return np.asarray(u, dtype=floatX)
    else:
        return np.asarray(v, dtype=floatX)
        
#################################################################    
def zeros(shape):
    return np.zeros(shape, dtype=floatX)
    
#################################################################
def apply_init(init_func, param):
    param.set_value(init_func(param.get_value().shape))