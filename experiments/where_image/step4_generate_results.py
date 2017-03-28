from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import configparser
import collections
import numpy as np
import scipy.stats
import time
import timeit
import json
import sys

from lib.langmod_tools import *
from lib.seqs import *
from lib.dates import *
from lib.files import *
from architecture_list import *

calc_prb = True
calc_gen = True
calc_div = True
calc_ret = True

cfg = configparser.ConfigParser()
cfg.read('config.ini')

raw_input_data_dir       = cfg.get('DIRS', 'RawInputDataDir')
mscoco_dir               = cfg.get('DIRS', 'MSCOCODir')
processed_input_data_dir = cfg.get('DIRS', 'ProcessedInputDataDir')
prb_generated_data_dir   = cfg.get('DIRS', 'PrbGeneratedDataDir')
gen_generated_data_dir   = cfg.get('DIRS', 'GenGeneratedDataDir')
ret_generated_data_dir   = cfg.get('DIRS', 'RetGeneratedDataDir')
results_dir              = cfg.get('DIRS', 'ResultsDir')
num_runs                 = cfg.getint('TRAIN', 'NumRuns')

create_dir(results_dir)

sys.path.append(mscoco_dir)
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

################################################################
print('============================================')
print('Loading processed data...')
print()
sys.stdout.flush()

with open(processed_input_data_dir+'/test_humancaptions.txt', 'r', encoding='utf-8') as f:
    test_humancaptions = [ caption.split(' ') for caption in f.read().strip().split('\n') ]

with open(processed_input_data_dir+'/vocabulary.txt', 'r', encoding='utf-8') as f:
    num_known_tokens = len(f.read().split('\n'))
    
################################################################
print('============================================')
print('Calculating results...')
print()
sys.stdout.flush()

################################################################
if calc_prb:
    print('============================================')
    print('Probability measures')
    print()
    sys.stdout.flush()
    with open(results_dir+'/results_prb.txt', 'w', encoding='utf-8') as f_out:
        print('architecture', 'rnn', 'run', 'pplx_geomean', 'pplx_artmean', 'pplx_median', 'prob_geomean', 'prob_artmean', 'prob_median', sep='\t', file=f_out)
    
        for run in range(1, num_runs+1):
            for (architecture_name, rnn_name) in testable_architectures:
                if not file_exists(prb_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run)):
                    print('SKIPPING (no data):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                    continue
                    
                run_start_time = timeit.default_timer()
                
                print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime("%Y/%m/%d %H:%M:%S")))
                sys.stdout.flush()
            
                with open(prb_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'r', encoding='utf-8') as f:
                    caption_probs = list()
                    caption_pplxs = list()
                    for line in f:
                        token_probs = [ float(x) for x in line.strip().split('\t') ]
                        
                        caption_prob = sequence_probability(token_probs)
                        caption_probs.append(caption_prob)

                        caption_pplx = sequence_perplexity(token_probs)
                        caption_pplxs.append(caption_pplx)

                prob_artmean = np.mean(caption_probs, dtype=np.float64)
                prob_geomean = scipy.stats.gmean(caption_probs, dtype=np.float64)
                prob_median  = np.median(caption_probs)
                
                pplx_artmean = np.mean(caption_pplxs, dtype=np.float64)
                pplx_geomean = scipy.stats.gmean(caption_pplxs, dtype=np.float64)
                pplx_median  = np.median(caption_pplxs)
                
                prb_result = [ pplx_geomean, pplx_artmean, pplx_median, prob_geomean, prob_artmean, prob_median ]
                print(*[ str(x) for x in [architecture_name, rnn_name, run]+prb_result ], sep='\t', file=f_out)
                
                run_end_time = timeit.default_timer()
                print(format_duration(round(run_end_time-run_start_time)))
                print()
                sys.stdout.flush()
        
################################################################
if calc_gen:
    print('============================================')
    print('Generation measures')
    print()
    sys.stdout.flush()
    with open(results_dir+'/results_gen.txt', 'w', encoding='utf-8') as f_out:
        print('architecture', 'rnn', 'run', 'cider', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'meteor', 'rougel', 'wrong_word_pos', sep='\t', file=f_out)
        
        for run in range(1, num_runs+1):
            for (architecture_name, rnn_name) in testable_architectures:
                if not file_exists(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run)):
                    print('SKIPPING (no data):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                    continue
                    
                run_start_time = timeit.default_timer()
                
                print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime("%Y/%m/%d %H:%M:%S")))
                sys.stdout.flush()
                
                with open(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'r', encoding='utf-8') as f:
                    generated_captions = f.read().strip().split('\n')
                    
                with open(mscoco_dir+'/results/generated_captions.json', 'w', encoding='utf-8') as f_out_tmp:
                    print(str(json.dumps([
                            {
                                'image_id': image_id,
                                'caption': caption
                            }
                            for (image_id, caption) in enumerate(generated_captions)
                        ])), file=f_out_tmp)
                        
                coco = COCO(mscoco_dir+'/annotations/captions.json')
                cocoRes = coco.loadRes(mscoco_dir+'/results/generated_captions.json')
                cocoEval = COCOEvalCap(coco, cocoRes)
                cocoEval.evaluate()
                
                gen_result = [ cocoEval.eval[metric] for metric in [ 'CIDEr', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L' ] ]
                print(*[ str(x) for x in [architecture_name, rnn_name, run]+gen_result ], sep='\t', file=f_out)
                
                run_end_time = timeit.default_timer()
                print(format_duration(round(run_end_time-run_start_time)))
                print()
                sys.stdout.flush()
        
################################################################
if calc_div:
    print('============================================')
    print('Diversity measures')
    print()
    sys.stdout.flush()
    with open(results_dir+'/results_div.txt', 'w', encoding='utf-8') as f_out:
        print('architecture', 'rnn', 'run', 'known_vocab_used', 'unigram_entropy', 'bigram_entropy', sep='\t', file=f_out)
        
        for run in range(1, num_runs+1):
            for (architecture_name, rnn_name) in testable_architectures:
                if not file_exists(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run)):
                    print('SKIPPING (no data):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                    continue
                    
                run_start_time = timeit.default_timer()
                
                print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime("%Y/%m/%d %H:%M:%S")))
                sys.stdout.flush()
                
                with open(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'r', encoding='utf-8') as f:
                    unigram_freqs = collections.defaultdict(lambda:0)
                    bigram_freqs  = collections.defaultdict(lambda:0)
                    for line in f:
                        caption = line.strip().split(' ')
                        for unigram in caption:
                            unigram_freqs[unigram] += 1
                        for bigram in get_bigrams(caption):
                            bigram_freqs[bigram] += 1
                        
                known_vocab_used = len(unigram_freqs) / num_known_tokens
                    
                unigram_freqs = np.array(list(unigram_freqs.values()))
                unigram_probs = unigram_freqs/unigram_freqs.sum()
                unigram_entropy = -(unigram_probs*np.log2(unigram_probs)).sum()
                
                bigram_freqs = np.array(list(bigram_freqs.values()))
                bigram_probs = bigram_freqs/bigram_freqs.sum()
                bigram_entropy = -(bigram_probs*np.log2(bigram_probs)).sum()
                
                div_result = [ known_vocab_used, unigram_entropy, bigram_entropy ]
                print(*[ str(x) for x in [architecture_name, rnn_name, run]+div_result ], sep='\t', file=f_out)
                
                run_end_time = timeit.default_timer()
                print(format_duration(round(run_end_time-run_start_time)))
                print()
                sys.stdout.flush()
        
        for (name, skip) in [ ('human-one', 5), ('human-all', 1) ]:
            run_start_time = timeit.default_timer()
            
            print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(1, '', name, time.strftime("%Y/%m/%d %H:%M:%S")))
            sys.stdout.flush()
            
            unigram_freqs = collections.defaultdict(lambda:0)
            bigram_freqs  = collections.defaultdict(lambda:0)
            for caption in test_humancaptions[::skip]:
                for unigram in caption:
                    unigram_freqs[unigram] += 1
                for bigram in get_bigrams(caption):
                    bigram_freqs[bigram] += 1
                
            known_vocab_used = len(unigram_freqs) / num_known_tokens
                
            unigram_freqs = np.array(list(unigram_freqs.values()))
            unigram_probs = unigram_freqs/unigram_freqs.sum()
            unigram_entropy = -(unigram_probs*np.log2(unigram_probs)).sum()
            
            bigram_freqs = np.array(list(bigram_freqs.values()))
            bigram_probs = bigram_freqs/bigram_freqs.sum()
            bigram_entropy = -(bigram_probs*np.log2(bigram_probs)).sum()
            
            div_result = [ known_vocab_used, unigram_entropy, bigram_entropy ]
            print(*[ str(x) for x in [name, '', 1]+div_result ], sep='\t', file=f_out)
            
            run_end_time = timeit.default_timer()
            print(format_duration(round(run_end_time-run_start_time)))
            print()
            sys.stdout.flush()
        
################################################################
if calc_ret:
    print('============================================')
    print('Retrieval measures')
    print()
    sys.stdout.flush()
    with open(results_dir+'/results_ret.txt', 'w', encoding='utf-8') as f_out:
        print('architecture', 'rnn', 'run', 'R@1', 'R@5', 'R@10', sep='\t', file=f_out)
    
        for run in range(1, num_runs+1):
            for (architecture_name, rnn_name) in testable_architectures:
                if architecture_name == 'langmodel':
                    continue
                if not file_exists(ret_generated_data_dir+'/{}_{}_{}.npy'.format(rnn_name, architecture_name, run)):
                    print('SKIPPING (no data):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
                    continue
                    
                run_start_time = timeit.default_timer()
                
                print('run {0:>2} - {1:<8} {2:<30} | {3}'.format(run, rnn_name, architecture_name, time.strftime("%Y/%m/%d %H:%M:%S")))
                sys.stdout.flush()
            
                with open(ret_generated_data_dir+'/{}_{}_{}.npy'.format(rnn_name, architecture_name, run), 'rb') as f:
                    captionimages_probs = np.load(f)
                (r1, r5, r10) = (0, 0, 0)
                num_captions = 0
                for (row_num, captionimage_probs) in enumerate(captionimages_probs):
                    target_index = row_num//5
                    ordered = sorted(range(len(captionimage_probs)), key=lambda i:captionimage_probs[i], reverse=True)
                    target_found = ordered.index(target_index)
                    if target_found < 1:
                        r1  += 1
                        r5  += 1
                        r10 += 1
                    elif target_found < 5:
                        r5  += 1
                        r10 += 1
                    elif target_found < 10:
                        r10 += 1
                    num_captions += 1
                
                ret_result = [ r1/num_captions, r5/num_captions, r10/num_captions ]
                print(*[ str(x) for x in [architecture_name, rnn_name, run]+ret_result ], sep='\t', file=f_out)
            
                run_end_time = timeit.default_timer()
                print(format_duration(round(run_end_time-run_start_time)))
                print()
                sys.stdout.flush()

print(' '*50, time.strftime('%Y/%m/%d %H:%M:%S'))
sys.stdout.flush()