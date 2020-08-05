"""
This file contains the hyperparameters that affect the search space and the search method. Hyperparameters that can be
experimented with are called via main's argument parser.
"""

import sys
if not '../' in sys.path:
    sys.path.append('../')
import os
from pathlib import Path
#from pytorch_model.Aggregation import NoAggregation, Addition, Concatenation
#from pytorch_model.Operation import *
#from pytorch_model.Blocks import *
from copy import copy

print('settings is here')
TASK = 'fashion-mnist'

# Use the blocks-based search space? Also drives initial number of layers of architectures and depth of
# initial layer
BLOCK_SEARCH_SPACE = False
# Allow skip connections?
SKIP_ALLOWED = False
# Training and search regimes -- fast for easy tasks, slower for complex tasks. Also drives PAD_METHOD
# and INTER_METHOD and data augmentation.
if TASK in ['mnist', 'fashion-mnist', 'stocks']:
    IS_COMPLEX = False
else:
    IS_COMPLEX = True

# Dummy training for networks? (for testing purposes)
DUMMY = False  # Change to False when not debugging
# Placeholder: do we use the performance predictor? (value set by main.py arguments):
PREDICT = None
# Paths
EXEC_PATH = Path(os.getcwd())
RUN_PATH, WTS_PATH, POPS_PATH, LOG_PATH, PRED_BATCH_PATH = None,None,None,None,None
PRED_BATCH_FILENAME = None
VAE_WTS_PATH = None
# Path to dataset
DATA_PATH = EXEC_PATH/Path(r"../datasets/clnn_data")  # valid for datasets that ship with Pytorch
# DATA_PATH = Path(r"G:/datasets/stock_prediction_n_dates")  # otherwise enter full path to data
# Constant to prevent division by zero
EPS = 1e-10


MAX_OUTDEG_INPUT = 999
MAX_INDEG = 2  # Do not change these values inconsiderately! Untested behaviour could occur.
MAX_OUTDEG_HIDDEN = 2  # Max outdegree of hidden nodes (i.e. all except input node (Node 0) and classifier node)

VAE_hms=dict(hidden = [32, 64, 128, 256,512],
    k_sizes=[1,1,3,3,3],
    strides=[1,2,2,2,2],
    padding=[1,1,1,1,1],
    outpadding=[0,0,0,1,1])


VAE_HPARAMS = dict(
    # Note: when using the integrated VAE/Predictor, all these keys will be renamed vae_*
    weights_path=None,
    max_depth=10,
    n_chars=29,
    # device='cuda',
    max_len=56,  # maximum absolute length of all strings produced, regardless of the number of layers in the arch.
                 # only use even numbers when using a CNN decoder. Set as close as possible as maximum number of
                 # productions that can be used with the specified max number of layers.
    num_samples=5,
    weighted_sampling=True,
    temperature=0.5,  # softmax temperature for sampling sentence lengths; 0 = uniform distribution
    layer_symbol='LAY',
    # data_size=1,  # how many sequences constitute an epoch
    batch_size=256,  # must be >= 2
    latent_size=256,  # increasing latent size doesn't help convergence (but helps performance predictions)
    enc_type='cnn',  # 'cnn' or 'gru' or 'bi_gru
    dec_type='both',  # 'cnn', 'gru' or 'both'
    drop_rate=0.,
    length_encoding=True,
    fc_dim=512,

    # RNN decoder parameters
    rnn_hidden=512,
    rnn_layers=3,
    epsilon_std=0.01,  # stdev for the random variable sampled from N(0, std). Critical for performance

    # training parameters
    max_steps=500,
    lr_ini=1e-4,
    lr_min=1e-6,
    lr_reduce_patience=1500,
    lr_reduce_factor=0.2,
    early_stop_patience=2500,
    kld_wt_zero_for_epochs=1000,
    kld_wt_one_at_epoch=2000,
    kld_zero_value=1.e-15,
    kld_full_value=0.5,
    num_workers=4,
    categorical_channel=10,
    cont_gamma= 30.,
    disc_gamma= 30.,
    alpha=30.,
    hidden = VAE_hms['hidden'],
    k_sizes=VAE_hms['k_sizes'],
    strides=VAE_hms['strides'],
    padding=VAE_hms['padding'],
    outpadding=VAE_hms['outpadding'],
    hidden_d = VAE_hms['hidden'][::-1],
    k_sizes_d =VAE_hms['k_sizes'][::-1],
    strides_d = VAE_hms['strides'][::-1],
    padding_d = VAE_hms['padding'][::-1],
    outpadding_d = VAE_hms['outpadding'][::-1],
    con_min_capacity = 0.,
    con_max_capacity = 25.,
    dis_min_capacity= 0.,
    dis_max_capacity= 25.,
    cont_iter=100,
    disc_iter=100,
    anneal_rate=3e-5,
    anneal_interval=100

)

# Performance predictor hyperparameters
PRED_HPARAMS = dict(
    # Note: when using the integrated VAE/Predictor, all these keys will be renamed pred_*
    model_type='ensemble',  # 'mlp', 'exu', 'both', 'ensemble'
    mixup=False,
    mixup_alpha = 0.4,
    mc_dropout=False,
    activation='relu',  # activation function in hidden layers only
    weights_path=None,
    warm_up_period=3,  # number of generations to wait for before starting to train the predictor
    max_depth=VAE_HPARAMS['max_depth'],
    in_dim=VAE_HPARAMS['latent_size'], 
    c_dim=VAE_HPARAMS['categorical_channel'], # should be the same value as VAE_HPARAMS['latent_size']
    hid_dims='2048,2048,2048,2048,1024',  # dimensions of the hidden fully-connected layers
    droprate_hidden=0.25,
    droprate_head=0.2,
    max_epochs=30,  # number of epochs per training round
    batch_sz=128,
    w_decay=1.e-5,
    lr_ini=1e-4,
    lr_reduce_on_plateau_factor=.5,  # only used in independent experiments, not in NAS
    lr_reduce_patience=100,  # only used in independent experiments, not in NAS
    lr_reduce_mult_factor=0.99,
    lr_min=1.e-6,
    val_set_pct=0.,  # set to >0. in independent experiments but set to 0. when running NAS (not enough data)
    clone_split_ratio=.67,  # proportion of clones to train
    batch_gen_n=5,  # The number of generations of samples kept in fitness batch to train predictor,
    num_workers=4,
)
