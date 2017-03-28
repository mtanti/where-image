from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

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
from lib.langmod_tools import *
from architecture_list import *

make_prb = True
make_gen = True
make_ret = True

cfg = configparser.ConfigParser()
cfg.read('config.ini')

processed_input_data_dir      = cfg.get('DIRS', 'ProcessedInputDataDir')
prb_generated_data_dir        = cfg.get('DIRS', 'PrbGeneratedDataDir')
gen_generated_data_dir        = cfg.get('DIRS', 'GenGeneratedDataDir')
ret_generated_data_dir        = cfg.get('DIRS', 'RetGeneratedDataDir')
beam_width                    = cfg.getint('GEN', 'BeamWidth')
clip_len                      = cfg.getint('GEN', 'ClipLen')
caption_image_prob_batch_size = cfg.getint('BATCHES', 'CaptionImageProbBatchSize')
num_runs                      = cfg.getint('TRAIN', 'NumRuns')

create_dir(prb_generated_data_dir)
create_dir(gen_generated_data_dir)
create_dir(ret_generated_data_dir)

################################################################
print('============================================')
print('Loading processed data...')
print()
sys.stdout.flush()

with open(processed_input_data_dir+'/test_grouped_prefixes.npy', 'rb') as f:
    test_grouped_prefixes = np.load(f)
with open(processed_input_data_dir+'/test_grouped_targets.npy', 'rb') as f:
    test_grouped_targets = np.load(f)
with open(processed_input_data_dir+'/test_grouped_images.npy', 'rb') as f:
    test_grouped_images = np.load(f)
    
with open(processed_input_data_dir+'/vocabulary.txt', 'r', encoding='utf-8') as f:
    tokens = f.read().strip().split('\n')
prefix_vocabulary = Vocabulary(tokens, pad_index=0, start_index=1, unknown_index=-1)
target_vocabulary = Vocabulary(tokens, end_index=0, unknown_index=-1)

################################################################
if make_prb:
    print('============================================')
    print('Generating token probabilities...')
    print()
    sys.stdout.flush()

    for run in range(1, num_runs+1):
        for (architecture_name, rnn_name) in testable_architectures:
            if file_exists(prb_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run)):
                print('SKIPPING (was ready):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                continue
            if not architecture_params_exist(architecture_name, rnn_name, run):
                print('SKIPPING (no params):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                continue
                
            run_start_time = timeit.default_timer()
            
            print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime("%Y/%m/%d %H:%M:%S")))
            sys.stdout.flush()

            a = get_architecture(architecture_name, rnn_name, run)
            probs_table = list()
            for (image, prefix_group, target_group) in zip(test_grouped_images, test_grouped_prefixes, test_grouped_targets):
                if architecture_name == 'langmodel':
                    probs = a.testf('probability')(prefix_group, target_group)[0]
                else:
                    probs = a.testf('probability')(image.repeat(len(prefix_group), axis=0), prefix_group, target_group)[0]
                probs_table.append(probs.tolist())
            with open(prb_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'w', encoding='utf-8') as f:
                for row in probs_table:
                    print(*[ str(p) for p in row ], sep='\t', file=f)
            
            run_end_time = timeit.default_timer()
            print(format_duration(round(run_end_time-run_start_time)))
            print()
            sys.stdout.flush()

################################################################
if make_gen:
    print('============================================')
    print('Generating captions...')
    print()
    sys.stdout.flush()

    for run in range(1, num_runs+1):
        for (architecture_name, rnn_name) in testable_architectures:
            if file_exists(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run)):
                print('SKIPPING (was ready):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                continue
            if not architecture_params_exist(architecture_name, rnn_name, run):
                print('SKIPPING (no params):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                continue
                
            run_start_time = timeit.default_timer()
            
            print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime("%Y/%m/%d %H:%M:%S")))
            sys.stdout.flush()
                                                
            a = get_architecture(architecture_name, rnn_name, run)
            if architecture_name == 'langmodel':
                predictions_function = lambda prefixes:a.testf('prediction')(prefixes)[0]
                tokens = generate_sequence_beamsearch(predictions_function, prefix_vocabulary, target_vocabulary, beam_width, None, clip_len)[0]
                caption = ' '.join(tokens)
                captions = [ caption ]*len(test_grouped_images[::5])
            else:
                captions = list()
                for image in test_grouped_images[::5]:
                    predictions_function = lambda prefixes:a.testf('prediction')(image.repeat(len(prefixes), axis=0), prefixes)[0]
                    tokens = generate_sequence_beamsearch(predictions_function, prefix_vocabulary, target_vocabulary, beam_width, None, clip_len)[0]
                    caption = ' '.join(tokens)
                    captions.append(caption)
            with open(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'w', encoding='utf-8') as f:
                print(str('\n'.join(captions)), file=f)
            
            run_end_time = timeit.default_timer()
            print(format_duration(round(run_end_time-run_start_time)))
            print()
            sys.stdout.flush()

################################################################
if make_ret:
    print('============================================')
    print('Generating image-caption probabilities...')
    print()
    sys.stdout.flush()

    for run in range(1, num_runs+1):
        for (architecture_name, rnn_name) in testable_architectures:
            if architecture_name == 'langmodel':
                continue
            if file_exists(ret_generated_data_dir+'/{}_{}_{}.npy'.format(rnn_name, architecture_name, run)):
                print('SKIPPING (was ready):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                continue
            if not architecture_params_exist(architecture_name, rnn_name, run):
                print('SKIPPING (no params):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                continue
                
            run_start_time = timeit.default_timer()
            
            print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime("%Y/%m/%d %H:%M:%S")))
            sys.stdout.flush()
                                                
            a = get_architecture(architecture_name, rnn_name, run)
            num_captions = len(test_grouped_targets)
            num_images = num_captions//5
            captionimages_probs = np.empty((num_captions, num_images), 'float64')
            for (col_num, image) in enumerate(test_grouped_images[::5]):
                for caption_batch in chunker(enumerate(zip(test_grouped_prefixes, test_grouped_targets)), caption_image_prob_batch_size):
                    joined_batch_prefixes = list()
                    joined_batch_targets  = list()
                    joined_batch_bounds   = list()
                    last_bound = 0
                    for (row_num, (prefix_group, target_group)) in caption_batch:
                        joined_batch_prefixes.extend(prefix_group)
                        joined_batch_targets.extend(target_group)
                        joined_batch_bounds.append((row_num, last_bound, last_bound+len(target_group)))
                        last_bound += len(target_group)
                    token_probs = a.testf('probability')(image.repeat(len(joined_batch_targets), axis=0), joined_batch_prefixes, joined_batch_targets)[0]
                    for (row_num, bound1, bound2) in joined_batch_bounds:
                        seq_prob = sequence_probability(token_probs[bound1:bound2])
                        captionimages_probs[row_num, col_num] = seq_prob
                 
            with open(ret_generated_data_dir+'/{}_{}_{}.npy'.format(rnn_name, architecture_name, run), 'wb') as f:
                np.save(f, captionimages_probs)

            run_end_time = timeit.default_timer()
            print(format_duration(round(run_end_time-run_start_time)))
            print()
            sys.stdout.flush()

print(' '*50, time.strftime('%Y/%m/%d %H:%M:%S'))
sys.stdout.flush()
