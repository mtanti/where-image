from __future__ import absolute_import, division, print_function, unicode_literals
from builtins import ascii, bytes, chr, dict, filter, hex, input, int, map, next, oct, open, pow, range, round, str, super, zip

import configparser
import json
import gzip

from architecture.architecture import *
from architecture.mods import *
from architecture.activations import *
from architecture.feedforward import *
from architecture.inputs import *
from architecture.joiners import *
from architecture.misc import *
from architecture.rnn import *
from architecture.constants import *
from architecture.optimizers import *

from lib.files import *

cfg = configparser.ConfigParser()
cfg.read('config.ini')

processed_input_data_dir = cfg.get('DIRS', 'ProcessedInputDataDir')
neural_net_params_dir    = cfg.get('DIRS', 'NeuralNetParamsDir')
training_costs_dir       = cfg.get('DIRS', 'TrainingCostsDir')
grad_clip                = cfg.getfloat('TRAIN', 'GradClip')
pad_index                = cfg.getint('VOCAB', 'PadIndex')
non_pad_index            = cfg.getint('VOCAB', 'NonPadIndex')
dropout_rate             = cfg.getfloat('ARCH', 'DropoutRate')
image_size               = cfg.getint('ARCH', 'ImageSize')
state_size               = cfg.getint('ARCH', 'StateSize')
embed_size               = cfg.getint('ARCH', 'EmbedSize')
max_epochs               = cfg.getint('TRAIN', 'MaxEpochs')
early_stop_patience      = cfg.getint('TRAIN', 'EarlyStopPatience')

create_dir(neural_net_params_dir)
create_dir(training_costs_dir)

################################################################

with open(processed_input_data_dir+'/vocabulary.txt', 'r', encoding='utf-8') as f:
    tokens = f.read().strip().split('\n')
input_vocab_size  = len(tokens) + 3 #pad, start, unknown
output_vocab_size = len(tokens) + 2 #end, unknown

optimizer = Adam(gradient_clipping_magnitude=grad_clip)

merge_training_mods = {
    'training': ([ 'image', 'prefix', 'target' ], [ 'output', 'target' ], sum_crossentropy, optimizer),
}
merge_testing_mods = {
    'validation':  ([ 'image', 'prefix', 'target' ], [ 'output', 'target' ], sum_crossentropy),
    'probability': ([ 'image', 'prefix', 'target' ], [ 'output', 'target' ], select_one),
    'prediction':  ([ 'image', 'prefix' ], [ 'output' ], no_mod),
    'embed':       ([ 'prefix' ], [ 'embed' ], no_mod),
    'rnn':         ([ 'prefix' ], [ 'rnn' ], no_mod),
    'image':       ([ 'image' ], [ 'image_dense' ], no_mod),
    'preoutput':  ([ 'image', 'prefix' ], [ 'preoutput_dense' ], no_mod),
}
inject_training_mods = {
    'training': ([ 'image', 'prefix', 'target' ], [ 'output', 'target' ], sum_crossentropy, optimizer),
}
inject_testing_mods = {
    'validation':  ([ 'image', 'prefix', 'target' ], [ 'output', 'target' ], sum_crossentropy),
    'probability': ([ 'image', 'prefix', 'target' ], [ 'output', 'target' ], select_one),
    'prediction':  ([ 'image', 'prefix' ], [ 'output' ], no_mod),
    'embed':       ([ 'prefix' ], [ 'embed' ], no_mod),
    'rnn':         ([ 'image', 'prefix' ], [ 'rnn' ], no_mod),
    'image':       ([ 'image' ], [ 'image_dense' ], no_mod),
    'preoutput':  ([ 'image', 'prefix' ], [ 'preoutput_dense' ], no_mod),
}
langmod_training_mods = {
    'training': ([ 'prefix', 'target' ], [ 'output', 'target' ], sum_crossentropy, optimizer),
}
langmod_testing_mods = {
    'validation':  ([ 'prefix', 'target' ], [ 'output', 'target' ], sum_crossentropy),
    'probability': ([ 'prefix', 'target' ], [ 'output', 'target' ], select_one),
    'prediction':  ([ 'prefix' ], [ 'output' ], no_mod),
    'embed':       ([ 'prefix' ], [ 'embed' ], no_mod),
    'rnn':         ([ 'prefix' ], [ 'rnn' ], no_mod),
    'preoutput':  ([ 'prefix' ], [ 'preoutput_dense' ], no_mod),
}

special_architecture_names = [
    'mao',
    'vinyals',
]

rnn_names = [
    'srnn',
    'lstm',
]

testable_architectures = [
    ('mao',          ''),
    ('vinyals',      ''),
    ('merge_concat', 'srnn'),
    ('merge_add',    'srnn'),
    ('merge_mult',   'srnn'),
    ('inject_post',  'srnn'),
    ('inject_par',   'srnn'),
    ('inject_pre',   'srnn'),
    ('inject_init',  'srnn'),
    ('langmodel',    'srnn'),
    ('merge_concat', 'lstm'),
    ('merge_add',    'lstm'),
    ('merge_mult',   'lstm'),
    ('inject_post',  'lstm'),
    ('inject_par',   'lstm'),
    ('inject_pre',   'lstm'),
    ('inject_init',  'lstm'),
    ('langmodel',    'lstm'),
]

################################################################
def get_architecture(architecture_name, rnn_name, run_to_load=None):
    if rnn_name == '' and architecture_name not in special_architecture_names or rnn_name != '' and architecture_name in special_architecture_names:
        raise ValueError('Cannot specify rnn for special architectures (only).')
    elif rnn_name == '':
        rnn = None
    elif rnn_name == 'srnn':
        rnn = SimpleRnn
    elif rnn_name == 'lstm':
        rnn = Lstm
    else:
        raise ValueError('Unknown rnn name.')

    #Select network architecture
    if architecture_name == 'mao':
        nn = Architecture(
                [
                    Softmax('output',
                        Dense('preoutput_dense', output_vocab_size,
                            ScaledTanh('merge_activation',
                                MergeAdd('merge',
                                    Dense('rnn_dense', 512,
                                        SimpleRnn('rnn', 256, pad_index,
                                            Zeros('initstate', 256,
                                                Ref('@prefix-initstate', 'prefix')
                                            ),
                                            Ref('@prefix-mask', 'prefix'),
                                            Dense('embed_dense', 256,
                                                Dropout('embed_drop', dropout_rate,
                                                    EmbeddedSeqs('embed', 256,
                                                        IndexSeqsInput('prefix', input_vocab_size)
                                                    )
                                                )
                                            )
                                        )
                                    ),
                                    MergeAdd('merge2',
                                        Dense('image_dense', 512,
                                            Dropout('image_drop', dropout_rate,
                                                VectorsInput('image', image_size)
                                            )
                                        ),
                                        Dense('lastword_dense', 512,
                                            LastItem('lastword',
                                                Ref('@embed_dense-lastword', 'embed_dense')
                                            )
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    IntsInput('target')
                ],
                merge_testing_mods,
                merge_training_mods,
            )
    elif architecture_name == 'vinyals':
        nn = Architecture(
                [
                    Softmax('output',
                        Dense('preoutput_dense', output_vocab_size,
                            Lstm('rnn', 512, pad_index,
                                Zeros('initstate', 512,
                                    Ref('@prefix-initstate', 'prefix')
                                ),
                                PreInjectMask('inject_mask', non_pad_index,
                                    Ref('@prefix-mask', 'prefix')
                                ),
                                PreInjectSeq('inject_seq',
                                    Dense('image_dense', 512,
                                        Dropout('image_drop', dropout_rate,
                                            VectorsInput('image', image_size)
                                        )
                                    ),
                                    Dropout('embed_drop', dropout_rate,
                                        EmbeddedSeqs('embed', 512,
                                            IndexSeqsInput('prefix', input_vocab_size)
                                        )
                                    )
                                )
                            )
                        )
                    ),
                    IntsInput('target')
                ],
                inject_testing_mods,
                inject_training_mods,
            )
    elif architecture_name.startswith('merge_'):
        if architecture_name.endswith('_concat'):
            merge = MergeConcat
        elif architecture_name.endswith('_add'):
            merge = MergeAdd
        elif architecture_name.endswith('_mult'):
            merge = MergeMult
        else:
            raise ValueError('Unknown merge type.')
        nn = Architecture(
                [
                    Softmax('output',
                        Dense('preoutput_dense', output_vocab_size,
                            merge('merge',
                                rnn('rnn', state_size, pad_index,
                                    Zeros('initstate', state_size,
                                        Ref('@prefix-initstate', 'prefix')
                                    ),
                                    Ref('@prefix-mask', 'prefix'),
                                    Dropout('embed_drop', dropout_rate,
                                        EmbeddedSeqs('embed', embed_size,
                                            IndexSeqsInput('prefix', input_vocab_size)
                                        )
                                    )
                                ),
                                Dense('image_dense', state_size,
                                    Dropout('image_drop', dropout_rate,
                                        VectorsInput('image', image_size)
                                    )
                                )
                            )
                        )
                    ),
                    IntsInput('target')
                ],
                merge_testing_mods,
                merge_training_mods,
            )
    elif architecture_name.startswith('inject_'):
        if architecture_name.endswith('_pre') or architecture_name.endswith('_post'):
            if architecture_name.endswith('_pre'):
                mask_inject = PreInjectMask
                seq_inject  = PreInjectSeq
            else:
                mask_inject = PostInjectMask
                seq_inject  = PostInjectSeq
            nn = Architecture(
                    [
                        Softmax('output',
                            Dense('preoutput_dense', output_vocab_size,
                                rnn('rnn', state_size, pad_index,
                                    Zeros('initstate', state_size,
                                        Ref('@prefix-initstate', 'prefix')
                                    ),
                                    mask_inject('inject_mask', non_pad_index,
                                        Ref('@prefix-mask', 'prefix')
                                    ),
                                    seq_inject('inject_seq',
                                        Dense('image_dense', state_size,
                                            Dropout('image_drop', dropout_rate,
                                                VectorsInput('image', image_size)
                                            )
                                        ),
                                        Dropout('embed_drop', dropout_rate,
                                            EmbeddedSeqs('embed', embed_size,
                                                IndexSeqsInput('prefix', input_vocab_size)
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        IntsInput('target')
                    ],
                    inject_testing_mods,
                    inject_training_mods,
                )
        elif architecture_name.endswith('_par'):
            nn = Architecture(
                    [
                        Softmax('output',
                            Dense('preoutput_dense', output_vocab_size,
                                rnn('rnn', state_size, pad_index,
                                    Zeros('initstate', state_size,
                                        Ref('@prefix-initstate', 'prefix')
                                    ),
                                    Ref('@prefix-mask', 'prefix'),
                                    ParInjectSeq('inject_seq',
                                        Dense('image_dense', state_size,
                                            Dropout('image_drop', dropout_rate,
                                                VectorsInput('image', image_size)
                                            )
                                        ),
                                        Dropout('embed_drop', dropout_rate,
                                            EmbeddedSeqs('embed', embed_size,
                                                IndexSeqsInput('prefix', input_vocab_size)
                                            )
                                        )
                                    )
                                )
                            )
                        ),
                        IntsInput('target')
                    ],
                    inject_testing_mods,
                    inject_training_mods,
                )
        elif architecture_name.endswith('_init'):
            nn = Architecture(
                    [
                        Softmax('output',
                            Dense('preoutput_dense', output_vocab_size,
                                rnn('rnn', state_size, pad_index,
                                    Dense('image_dense', state_size,
                                        Dropout('image_drop', dropout_rate,
                                            VectorsInput('image', image_size)
                                        )
                                    ),
                                    Ref('@prefix-mask', 'prefix'),
                                    Dropout('embed_drop', dropout_rate,
                                        EmbeddedSeqs('embed', embed_size,
                                            IndexSeqsInput('prefix', input_vocab_size)
                                        )
                                    )
                                )
                            )
                        ),
                        IntsInput('target')
                    ],
                    inject_testing_mods,
                    inject_training_mods,
                )
        else:
            raise ValueError('Unknown inject type.')
    elif architecture_name == 'langmodel':
        nn = Architecture(
                [
                    Softmax('output',
                        Dense('preoutput_dense', output_vocab_size,
                            rnn('rnn', state_size, pad_index,
                                Zeros('initstate', state_size,
                                    Ref('@prefix-initstate', 'prefix')
                                ),
                                Ref('@prefix-mask', 'prefix'),
                                Dropout('prefix_drop', dropout_rate,
                                    EmbeddedSeqs('embed', embed_size,
                                        IndexSeqsInput('prefix', input_vocab_size)
                                    )
                                )
                            )
                        )
                    ),
                    IntsInput('target')
                ],
                langmod_testing_mods,
                langmod_training_mods,
            )
    else:
        raise ValueError('Unknown architecture name.')
        
    #Compile network
    nn.compile()
    
    #Load saved parameters if requested
    if run_to_load is not None:
        with gzip.open(neural_net_params_dir+'/{}_{}_{}.json.gz'.format(rnn_name, architecture_name, run_to_load), 'rb') as f:
            params = json.loads(str(f.read(), 'ascii'))
        nn.set_params(params)
    
    return nn
    
################################################################
def save_architecture_params(architecture_name, rnn_name, run_to_save, architecture):
    with gzip.open(neural_net_params_dir+'/{}_{}_{}.json.gz'.format(rnn_name, architecture_name, run_to_save), 'wb') as f:
        f.write(bytes(json.dumps(architecture.get_params()), 'ascii'))

################################################################
def save_epoch_info(architecture_name, rnn_name, run, epoch, training_cost, validation_cost, new_best):
    if epoch == 0:
        with open(training_costs_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'w', encoding='utf-8') as f:
            print(*[ str(x) for x in ['', 'training', 'validation'] ], sep='\t', file=f)
    
    with open(training_costs_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'a', encoding='utf-8') as f:
        print(*[ str(x) for x in [epoch, training_cost, validation_cost] ], sep='\t', file=f)
            
################################################################
def architecture_params_exist(architecture_name, rnn_name, run):
    '''Check if a fully trained parameters file exists for a requested architecture. If file exists then this function will check the training costs file to check that a termination criteria has been reached by early stopping.'''
    if file_exists(neural_net_params_dir+'/{}_{}_{}.json.gz'.format(rnn_name, architecture_name, run)):
        with open(training_costs_dir+'/{}_{}_{}.txt'.format(rnn_name, architecture_name, run), 'r', encoding='utf-8') as f:
            validation_costs = [ float(validation_cost) for (epoch, training_cost, validation_cost) in [ line.split('\t') for line in f.read().strip().split('\n')[1:] ] ]
        if len(validation_costs) == max_epochs + 1:
            return True
        final_validation_costs = validation_costs[-(early_stop_patience+1):]
        if len(final_validation_costs) == early_stop_patience+1:
            return all(final_validation_costs[0] < other_cost for other_cost in final_validation_costs[1:])
        else:
            return False
    else:
        return False
