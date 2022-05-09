import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from clu import platform

import os

data_config = {'data_dir': "data",
              'mask_path': "data/masks/COM_Mask_CMB-common-Mask-Int_2048_R3.fits",
              'nside': 128,
              'indices': 'adaptive',
              'reduction_factor': None,
              'output_shape': 'valid_only',
              'trainperc': 0.75,
              'evalperc': 0.01}

training_config = {'num_epochs': 20,
                   'warmup_epochs': 1,
                   'batch_size': 2,
                   'init_lr': 0.2,
                   'half_precision': False,
                   'momentum': 0.8,
                   'nesterov': True,
                   'workdir': "runs/run_graph_vit_1",
                   'log_every_step': 100}

#model configs:
config_cnn = {'dropout_rate': 0.1, 
              'res_depth': 3,
              'include_top': True}

config_mixer = {'nside_super': 4,
                'K_p': 4,
                'conv_depth': 2,
                'conv_features': 128,
                'num_mixer_blocks': 6,
                'hidden_dim': 192,
                'tokens_mlp_dim': 128,
                'channels_mlp_dim': 384,
                'superpix_threshold': 0.5,
                'include_top': True}

config_graph_vit = {'nside_super': 4,
                    'K_p': 4,
                    'conv_depth': 2,
                    'conv_features': 64,
                    'hidden_dim': 128,
                    'positional_embedding': True,
                    'num_encoder_blocks': 6,
                    'num_heads': 4,
                    'mlp_dim': 192,
                    'dropout_rate': 0.05,
                    'attention_dropout_rate': 0.05,
                    'superpix_mask_threshold': 0.5, 
                    'include_top': True,
                    'classifier': 'token'}

config_graph_vit_reg_attn = {'nside_super': 4,
                    'K_p': 4,
                    'conv_depth': 2,
                    'conv_features': 96,
                    'hidden_dim': 184,
                    'positional_embedding': True,
                    'num_encoder_blocks': 4,
                    'num_heads': 4,
                    'mlp_dim': 128,
                    'dropout_rate': 0.1,
                    'attention_dropout_rate': 0.1, 
                    'include_top': True}

