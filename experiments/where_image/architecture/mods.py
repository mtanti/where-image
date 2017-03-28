from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import theano.tensor as T
import math
import numpy as np

from architecture.layer import *

floatX = theano.config.floatX

##################################################################################################################################
def no_mod(*models):
    return list(models)
        
##################################################################################################################################
def sum_crossentropy(softmaxes, targets):
    return [ T.sum(T.nnet.categorical_crossentropy(softmaxes, targets)) ]
    
##################################################################################################################################
def select_one(softmaxes, indexes):
    return [ softmaxes[T.arange(softmaxes.shape[0]), indexes] ]
    