from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import theano
import configparser
import numpy as np
import random
import json
import gzip
import time
import timeit
import sys

from lib.vocabulary import *
from lib.seqs import *
from lib.dates import *
from lib.files import *
from architecture_list import *

cfg = configparser.ConfigParser()
cfg.read('config.ini')

processed_input_data_dir = cfg.get('DIRS', 'ProcessedInputDataDir')
num_runs                 = cfg.getint('TRAIN', 'NumRuns')
max_epochs               = cfg.getint('TRAIN', 'MaxEpochs')
validation_batch_size    = cfg.getint('BATCHES', 'ValidationBatchSize')
training_minibatch_size  = cfg.getint('TRAIN', 'MinibatchSize')
early_stop_patience      = cfg.getint('TRAIN', 'EarlyStopPatience')

################################################################
print('Loading processed data...')
sys.stdout.flush()

with open(processed_input_data_dir+'/training_prefixes.npy', 'rb') as f:
    training_prefixes = np.load(f)
with open(processed_input_data_dir+'/training_targets.npy', 'rb') as f:
    training_targets = np.load(f)
with open(processed_input_data_dir+'/training_indexes.npy', 'rb') as f:
    training_indexes = np.load(f)
with open(processed_input_data_dir+'/training_images.npy', 'rb') as f:
    training_images = np.load(f)

with open(processed_input_data_dir+'/validation_prefixes.npy', 'rb') as f:
    validation_prefixes = np.load(f)
with open(processed_input_data_dir+'/validation_targets.npy', 'rb') as f:
    validation_targets = np.load(f)
with open(processed_input_data_dir+'/validation_indexes.npy', 'rb') as f:
    validation_indexes = np.load(f)
with open(processed_input_data_dir+'/validation_images.npy', 'rb') as f:
    validation_images = np.load(f)

with open(processed_input_data_dir+'/vocabulary.txt', 'r', encoding='utf-8') as f:
    tokens = f.read().strip().split('\n')
prefix_vocabulary = Vocabulary(tokens, pad_index=0, start_index=1, unknown_index=-1)
target_vocabulary = Vocabulary(tokens, end_index=0, unknown_index=-1)

################################################################
print('Training...')
print()
sys.stdout.flush()

training_minibatch_index_list = list(range(len(training_prefixes)))
validation_batch_index_list = list(range(len(validation_prefixes)))
for run in range(1, num_runs+1):
    for (architecture_name, rnn_name) in testable_architectures:
        if architecture_params_exist(architecture_name, rnn_name, run):
            print('SKIPPING (was ready):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
            continue
            
        run_start_time = timeit.default_timer()
    
        print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime('%Y/%m/%d %H:%M:%S')))
        sys.stdout.flush()
        
        a = get_architecture(architecture_name, rnn_name)
        best_cost = None
        epochs_since_last_best = 0
        for epoch in range(0, max_epochs+1):
            epoch_start_time = timeit.default_timer()
            
            #Train epoch
            if epoch > 0:
                print('    epoch {0:>2} - training... '.format(epoch), end='')
                
                random.shuffle(training_minibatch_index_list)
                total = 0
                for minibatch in chunker(training_minibatch_index_list, training_minibatch_size):
                    if architecture_name == 'langmodel':
                        total += a.trainf('training')(training_prefixes[minibatch], training_targets[minibatch])[0]
                    else:
                        total += a.trainf('training')(training_images[training_indexes[minibatch]], training_prefixes[minibatch], training_targets[minibatch])[0]
                training_cost = total/len(training_prefixes)
                print('{0:>10.3f} | validating... '.format(training_cost), end='')
            else:
                print('    epoch {0:>2} - {1} | validating... '.format(epoch, ' '*(12+10)), end='')
                training_cost = ''
            sys.stdout.flush()
            
            #Validate epoch
            total = 0
            for minibatch in chunker(validation_batch_index_list, validation_batch_size):
                if architecture_name == 'langmodel':
                    total += a.testf('validation')(validation_prefixes[minibatch], validation_targets[minibatch])[0]
                else:
                    total += a.testf('validation')(validation_images[validation_indexes[minibatch]], validation_prefixes[minibatch], validation_targets[minibatch])[0]
            validation_cost = total/len(validation_prefixes)
            
            epoch_end_time = timeit.default_timer()
            print('{0:>10.3f} {1:8} | {2} {3}'.format(validation_cost, 'NEW BEST' if epoch > 0 and validation_cost < best_cost else '', time.strftime('%H:%M:%S'), format_duration(round(epoch_end_time-epoch_start_time))))
            sys.stdout.flush()
            
            #Post epoch processes
            save_epoch_info(architecture_name, rnn_name, run, epoch, training_cost, validation_cost, epoch > 0 and validation_cost < best_cost)
            if epoch == 0 or validation_cost < best_cost:
                epochs_since_last_best = 0
                best_cost = validation_cost
                save_architecture_params(architecture_name, rnn_name, run, a)
            else:
                epochs_since_last_best += 1
            if epochs_since_last_best >= early_stop_patience:
                break

        run_end_time = timeit.default_timer()
        print(format_duration(round(run_end_time-run_start_time)))
        print()
        sys.stdout.flush()
            
print(' '*50, time.strftime('%Y/%m/%d %H:%M:%S'))
sys.stdout.flush()
