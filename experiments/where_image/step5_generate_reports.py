from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import configparser
import collections
import numpy as np
import heapq
import time
import timeit
import json
import re
import sys

from lib.dates import *
from lib.files import *
from architecture_list import *

cfg = configparser.ConfigParser()
cfg.read('config.ini')

images_dir               = cfg.get('DIRS', 'ImagesDir')
processed_input_data_dir = cfg.get('DIRS', 'ProcessedInputDataDir')
gen_generated_data_dir   = cfg.get('DIRS', 'GenGeneratedDataDir')
ret_generated_data_dir   = cfg.get('DIRS', 'RetGeneratedDataDir')
results_dir              = cfg.get('DIRS', 'ResultsDir')
reports_dir              = cfg.get('DIRS', 'ReportsDir')
gen_demo_dir             = cfg.get('DIRS', 'GenDemoDir')
ret_demo_dir             = cfg.get('DIRS', 'RetDemoDir')
num_runs                 = cfg.getint('TRAIN', 'NumRuns')

create_dir(reports_dir)
create_dir(gen_demo_dir)
create_dir(ret_demo_dir)

################################################################
print('============================================')
print('Loading processed data...')
print()
sys.stdout.flush()

with open(processed_input_data_dir+'/test_humancaptions.txt', 'r', encoding='utf-8') as f:
    test_humancaptions = f.read().strip().split('\n')
with open(processed_input_data_dir+'/test_image_filenames.txt', 'r', encoding='utf-8') as f:
    image_fnames = f.read().strip().split('\n')

################################################################
print('============================================')
print('Producing reports...')
sys.stdout.flush()

reports_params = [
        {
            'data_file':    'results_prb.txt',
            'cols':         [ 'pplx_geomean', 'pplx_artmean', 'pplx_median' ],
            'title':        'PROBABILITY MEASURES',
            'sort_col':     'pplx_geomean',
            'sort_reverse': False,
            'sort_bottom':  set(),
            'best_func':    lambda means,col:min(means[col].values()),
            'col_format':   lambda col: '.3f'
        },
        {
            'data_file':    'results_gen.txt',
            'cols':         [ 'cider', 'meteor', 'rougel' ],
            'title':        'GENERATION MEASURES 1',
            'sort_col':     'cider',
            'sort_reverse': True,
            'sort_bottom':  set(),
            'best_func':    lambda means,col:max(means[col].values()),
            'col_format':   lambda col: '.3f'
        },
        {
            'data_file':    'results_gen.txt',
            'cols':         [ 'bleu1', 'bleu2', 'bleu3', 'bleu4' ],
            'title':        'GENERATION MEASURES 2',
            'sort_col':     'bleu1',
            'sort_reverse': True,
            'sort_bottom':  set(),
            'best_func':    lambda means,col:max(means[col].values()),
            'col_format':   lambda col: '.3f'
        },
        {
            'data_file':    'results_div.txt',
            'cols':         [ 'known_vocab_used', 'unigram_entropy', 'bigram_entropy' ],
            'title':        'DIVERSITY MEASURES',
            'sort_col':     'known_vocab_used',
            'sort_reverse': True,
            'sort_bottom':  { 'human-one', 'human-all' },
            'best_func':    lambda means,col:max(means[col][exp] for exp in means[col] if not exp.startswith('human')),
            'col_format':   lambda col: '.3%' if col == 'known_vocab_used' else '.3f'
        },
        {
            'data_file':    'results_ret.txt',
            'cols':         [ 'R@1', 'R@5', 'R@10' ],
            'title':        'RETRIEVAL MEASURES',
            'sort_col':     'R@1',
            'sort_reverse': True,
            'sort_bottom':  set(),
            'best_func':    lambda means,col:max(means[col].values()),
            'col_format':   lambda col: '.3%'
        },
    ]

run_start_time = timeit.default_timer()
with open(reports_dir+'/report_full.txt', 'w', encoding='utf-8') as f_out_full:
    for report_params in reports_params:
        with open(results_dir+'/'+report_params['data_file'], 'r', encoding='utf-8') as f:
            cols = report_params['cols']
            
            print(report_params['title'], file=f_out_full)
            print('experiment', *cols, sep='\t', file=f_out_full)
            
            exps = set()
            rows = [ line.split('\t') for line in f.read().strip().split('\n') ]
            all_cols = rows[0][3:]
            rows = rows[1:]
            data = { col: collections.defaultdict(list) for col in all_cols }
            if len(rows) == 0:
                continue
            for row in rows:
                (architecture_name, rnn_name, run) = row[:3]
                fields = row[3:]
                experiment = architecture_name + (('_'+rnn_name) if rnn_name != '' else '')
                for (col, field) in zip(all_cols, fields):
                    data[col][experiment].append(float(field))
                exps.add(experiment)
            means  = { col: { exp: np.mean(data[col][exp]) for exp in exps } for col in cols }
            stds   = { col: { exp: np.std(data[col][exp]) for exp in exps } for col in cols }
            bests  = { col: report_params['best_func'](means, col) for col in cols }
            order  = sorted(exps, key=lambda exp:(
                                                    (exp in report_params['sort_bottom']) != report_params['sort_reverse'],
                                                    means[report_params['sort_col']][exp]
                                                ), reverse=report_params['sort_reverse'])
            marker = lambda col,exp:'*' if means[col][exp] == bests[col] else ''
            
            for exp in order:
                print(*[ str(x) for x in [exp]+[
                        ('{0}{1:'+report_params['col_format'](col)+'} ({2:'+report_params['col_format'](col)+'})').format(marker(col, exp), means[col][exp], stds[col][exp])
                        for col in cols
                    ] ], sep='\t', file=f_out_full)
            print('', file=f_out_full)

with open(reports_dir+'/report_mini.txt', 'w', encoding='utf-8') as f_out_mini:
    with open(reports_dir+'/report_full.txt', 'r', encoding='utf-8') as f_out_full:
        for line in f_out_full:
            if line == '\n' or line.startswith('experiment\t') or line.startswith('PROBABILITY ') or line.startswith('GENERATION ') or line.startswith('DIVERSITY ') or line.startswith('RETRIEVAL '):
                print(line.strip(), file=f_out_mini)
            else:
                print(re.sub(r'\*| \([0-9.%E+-]*\)', '', line.strip()), file=f_out_mini)
            
for rnn_name in rnn_names:
    with open(reports_dir+'/report_mini_'+rnn_name+'.txt', 'w', encoding='utf-8') as f_out_mini_rnn:
        with open(reports_dir+'/report_mini.txt', 'r', encoding='utf-8') as f_out_mini:
            for line in f_out_mini:
                if line == '\n' or line.startswith('experiment\t') or line.startswith('PROBABILITY ') or line.startswith('GENERATION ') or line.startswith('DIVERSITY ') or line.startswith('RETRIEVAL '):
                    print(line.strip(), file=f_out_mini_rnn)
                else:
                    if line.split('\t')[0].endswith('_'+rnn_name):
                        print(line.strip(), file=f_out_mini_rnn)
with open(reports_dir+'/report_mini_-.txt', 'w', encoding='utf-8') as f_out_mini_special:
    with open(reports_dir+'/report_mini.txt', 'r', encoding='utf-8') as f_out_mini:
        for line in f_out_mini:
            if line == '\n' or line.startswith('experiment\t') or line.startswith('PROBABILITY ') or line.startswith('GENERATION ') or line.startswith('DIVERSITY ') or line.startswith('RETRIEVAL '):
                print(line.strip(), file=f_out_mini_special)
            else:
                if '_' not in line.split('\t')[0]:
                    print(line.strip(), file=f_out_mini_special)
    
with open(reports_dir+'/report_mini_stdev.txt', 'w', encoding='utf-8') as f_out_mini_stdev:
    with open(reports_dir+'/report_full.txt', 'r', encoding='utf-8') as f_out_full:
        for line in f_out_full:
            if line == '\n' or line.startswith('experiment\t') or line.startswith('PROBABILITY ') or line.startswith('GENERATION ') or line.startswith('DIVERSITY ') or line.startswith('RETRIEVAL '):
                print(line.strip(), file=f_out_mini_stdev)
            else:
                print(re.sub(r'\*?[0-9.%E+-]* \(|\)', '', line.strip()), file=f_out_mini_stdev)

    
run_end_time = timeit.default_timer()
print(format_duration(round(run_end_time-run_start_time)))
print()
sys.stdout.flush()

################################################################
print('============================================')
print('Generating gen demos...')
print()
sys.stdout.flush()

run_start_time = timeit.default_timer()

generated = dict()
for run in range(1, num_runs+1):
    for (architecture_name, rnn_name) in testable_architectures:
        if not file_exists(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run)):
            print('SKIPPING (no data):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
            continue
        with open(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'r', encoding='utf-8') as f_captions:
            generated[(rnn_name, architecture_name, run)] = f_captions.read().strip().split('\n')
        
        with open(gen_demo_dir+'/gen_{}_{}_{}.html'.format(rnn_name, architecture_name, run), 'w', encoding='utf-8') as f_out:
            print('<!DOCTYPE html>', file=f_out)
            print('<html>', file=f_out)
            print('<head>', file=f_out)
            print('    <title>gen - {} {} {}</title>'.format(rnn_name, architecture_name, run), file=f_out)
            print('    <style>', file=f_out)
            print('        img { max-width:300px; max-height:300px; width:auto; height:auto; }', file=f_out)
            print('        p   { font-family:arial; font-size:16pt; }', file=f_out)
            print('    </style>', file=f_out)
            print('    <base href="{}">'.format(images_dir+'/'), file=f_out)
            print('</head>', file=f_out)
            print('<body>', file=f_out)
            print('    <h1>gen - {} {} {}</h1>'.format(rnn_name, architecture_name, run), file=f_out)
            for (i, (image_fname, caption)) in enumerate(zip(image_fnames, generated[(rnn_name, architecture_name, run)])):
                    print('    <hr />', file=f_out)
                    print('    <h2 id="{0}"><a href="#{0}">#{0}</a></h2>'.format(i+1), file=f_out)
                    print('    <img src="{}" />'.format(image_fname), file=f_out)
                    print('    <p>{}</p>'.format(caption), file=f_out)
                    print('', file=f_out)
            print('</body>', file=f_out)
            print('</html>', file=f_out)
        
with open(gen_demo_dir+'/gen.html', 'w', encoding='utf-8') as f_out:
    print('<!DOCTYPE html>', file=f_out)
    print('<html>', file=f_out)
    print('<head>', file=f_out)
    print('    <title>gen - all</title>', file=f_out)
    print('    <style>', file=f_out)
    print('        img { max-width:300px; max-height:300px; width:auto; height:auto; }', file=f_out)
    print('        th  { font-family:arial; font-size:12pt; }', file=f_out)
    print('        td  { font-family:arial; font-size:16pt; }', file=f_out)
    print('    </style>', file=f_out)
    print('    <base href="{}">'.format(images_dir+'/'), file=f_out)
    print('</head>', file=f_out)
    print('<body>', file=f_out)
    print('    <h1>gen - all</h1>'.format(rnn_name, architecture_name, run), file=f_out)
    for (i, image_fname) in enumerate(image_fnames):
        print('    <hr />', file=f_out)
        print('    <h2 id="{0}"><a href="#{0}">#{0}</a></h2>'.format(i+1), file=f_out)
        print('    <img src="{}" />'.format(image_fname), file=f_out)
        print('    <table border="1">', file=f_out)
        for (architecture_name, rnn_name) in testable_architectures:
            for run in range(1, num_runs+1):
                if not file_exists(gen_generated_data_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run)):
                    continue
                print('        <tr>', file=f_out)
                print('            <th>{} {} {}</td>'.format(rnn_name, architecture_name, run), file=f_out)
                print('            <td>{}</td>'.format(generated[(rnn_name, architecture_name, run)][i]), file=f_out)
                print('        </tr>', file=f_out)
        print('    </table>', file=f_out)
        print('', file=f_out)
    print('</body>', file=f_out)
    print('</html>', file=f_out)
    
run_end_time = timeit.default_timer()
print(format_duration(round(run_end_time-run_start_time)))
print()
sys.stdout.flush()

################################################################
print('============================================')
print('Generating ret demos...')
print()
sys.stdout.flush()

run_start_time = timeit.default_timer()

retrieved = dict()
for run in range(1, num_runs+1):
    for (architecture_name, rnn_name) in testable_architectures:
        if architecture_name == 'langmodel':
            continue
        if not file_exists(ret_generated_data_dir+'/{}_{}_{}.npy'.format(rnn_name, architecture_name, run)):
            print('SKIPPING (no data):', 'run {0:>2} - {1:<8} {2:<30}'.format(run, rnn_name, architecture_name))
            continue
        
        with open(ret_generated_data_dir+'/{}_{}_{}.npy'.format(rnn_name, architecture_name, run), 'rb') as f:
            retrieved[(rnn_name, architecture_name, run)] = [ heapq.nlargest(3, range(len(row)), key=lambda i:row[i]) for row in np.load(f) ]
                
        with open(ret_demo_dir+'/ret_{}_{}_{}.html'.format(rnn_name, architecture_name, run), 'w', encoding='utf-8') as f_out:
            print('<!DOCTYPE html>', file=f_out)
            print('<html>', file=f_out)
            print('<head>', file=f_out)
            print('    <title>ret - {} {} {}</title>'.format(rnn_name, architecture_name, run), file=f_out)
            print('    <style>', file=f_out)
            print('        img{ max-width:200px; max-height:200px; width:auto; height:auto; }', file=f_out)
            print('        p  { font-family:arial; font-size:16pt; }', file=f_out)
            print('    </style>', file=f_out)
            print('    <base href="{}">'.format(images_dir+'/'), file=f_out)
            print('</head>', file=f_out)
            print('<body>', file=f_out)
            print('    <h1>ret - {} {} {}</h1>'.format(rnn_name, architecture_name, run), file=f_out)
            
            for (i, (caption, image_indexes)) in enumerate(zip(test_humancaptions, retrieved[(rnn_name, architecture_name, run)])):
                print('    <hr />', file=f_out)
                print('    <h2 id="{0}"><a href="#{0}">#{0}</a></h2>'.format(i+1), file=f_out)
                print('    <p>{}</p>'.format(caption.strip()), file=f_out)
                for image_i in image_indexes:
                    print('    <img src="{}" />'.format(image_fnames[image_i]), file=f_out)
                print('', file=f_out)
            print('</body>', file=f_out)
            print('</html>', file=f_out)
                
with open(ret_demo_dir+'/ret.html', 'w', encoding='utf-8') as f_out:
    print('<!DOCTYPE html>', file=f_out)
    print('<html>', file=f_out)
    print('<head>', file=f_out)
    print('    <title>ret - all</title>', file=f_out)
    print('    <style>', file=f_out)
    print('        img{ max-width:200px; max-height:200px; width:auto; height:auto; }', file=f_out)
    print('        th { font-family:arial; font-size:12pt; }', file=f_out)
    print('        p  { font-family:arial; font-size:16pt; }', file=f_out)
    print('    </style>', file=f_out)
    print('    <base href="{}">'.format(images_dir+'/'), file=f_out)
    print('</head>', file=f_out)
    print('<body>', file=f_out)
    print('    <h1>ret - all</h1>'.format(rnn_name, architecture_name, run), file=f_out)
    
    for (i, caption) in enumerate(test_humancaptions):
        print('    <hr />', file=f_out)
        print('    <h2 id="{0}"><a href="#{0}">#{0}</a></h2>'.format(i+1), file=f_out)
        print('    <p>{}</p>'.format(caption.strip()), file=f_out)
        print('    <table border="1">', file=f_out)
        for (architecture_name, rnn_name) in testable_architectures:
            for run in range(1, num_runs+1):
                if architecture_name == 'langmodel':
                    continue
                if not file_exists(ret_generated_data_dir+'/{}_{}_{}.npy'.format(rnn_name, architecture_name, run)):
                    continue
                print('        <tr>', file=f_out)
                print('            <th>{} {} {}</td>'.format(rnn_name, architecture_name, run), file=f_out)
                print('            <td>', file=f_out)
                for image_i in retrieved[(rnn_name, architecture_name, run)][i]:
                    print('                <img src="{}" />'.format(image_fnames[image_i]), file=f_out)
                print('            </td>', file=f_out)
                print('        </tr>', file=f_out)
        print('    </table>', file=f_out)
        print('', file=f_out)
    print('</body>', file=f_out)
    print('</html>', file=f_out)
                
run_end_time = timeit.default_timer()
print(format_duration(round(run_end_time-run_start_time)))
print()

print(' '*50, time.strftime('%Y/%m/%d %H:%M:%S'))
sys.stdout.flush()
