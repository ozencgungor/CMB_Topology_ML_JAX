import jax

from absl import logging
from clu import platform

import sys

from src import processtools as pt
from src import nngcn

import training_loop

import input_pipeline

import models_cnn
import models_graph_mixer
import models_graph_vit

from configs import config

import tensorflow as tf
#hide gpus from tensorflow:
tf.config.set_visible_devices([], 'GPU')

#load the config dicts:
data_config = config.data_config
training_config = config.training_config

#other model config files are also in the same config file.
#parameters like `nside`, `indices`, and `num_classes` are not in the config dictionaries
#but are rather retrieved from the dataset container class.
model_config = config.config_mixer

#dataset generation/input pipeline:
dataset = input_pipeline.dataset(data_dir=data_config['data_dir'],
                                 mask_path=data_config['mask_path'],
                                 nside=data_config['nside'],
                                 reduction_factor=data_config['reduction_factor'],
                                 trainperc=data_config['trainperc'],
                                 evalperc=data_config['evalperc'])

dataset.prepare_maps(indices=data_config['indices'], output_shape=data_config['output_shape'])

(train_ds, test_ds) = dataset.create_tf_datasets(global_batch_size=training_config['batch_size']*jax.device_count(),
                                                 prefetch=2)

#model definition/instantiation:
model = models_graph_mixer.Transformer(nside=dataset.nside,
                                 indices=dataset.adaptive_pix,
                                 nside_super=model_config['nside_super'], 
                                 K_p=model_config['K_p'], 
                                 conv_depth=model_config['conv_depth'],
                                 conv_features=model_config['conv_features'],
                                 hidden_dim=model_config['hidden_dim'],
                                 num_mixer_blocks=model_config['num_mixer_blocks'],
                                 tokens_mlp_dim=model_config['tokens_mlp_dim'],
                                 channels_mlp_dim=model_config['channels_mlp_dim'],
                                 superpix_threshold=model_config['superpix_threshold'],
                                 include_top=model_config['include_top'],
                                 num_classes=4)

def main():
    logging.set_verbosity(logging.INFO)
    logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
    logging.info('JAX local devices: %r', jax.local_devices())

    # Add a note so that we can tell which task is which JAX host.
    # (Depending on the platform task 0 is not guaranteed to be host 0)
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                         f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                         'runs/', 'workdir')
    
    final_state = training_loop.train_and_test(dataset=dataset, train_data=train_ds, 
                                               test_data=test_ds, model=model, 
                                               config=training_config)
    
    return 0 #just for sys.exit purposes.

if __name__ == '__main__': #execute the training loop.
    sys.exit(main())

