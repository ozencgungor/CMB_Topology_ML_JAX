import numpy as np
import matplotlib.pyplot as plt

import time

import tensorflow_datasets as tfds
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from tqdm import tqdm, trange

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.optim import dynamic_scale as dynamic_scale_lib
from flax.training import checkpoints, common_utils, train_state

import functools
from functools import partial

import ml_collections
import optax

from absl import logging
from clu import metric_writers, periodic_actions, platform
import os
import importlib

from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import input_pipeline

from configs import config

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]
PaddingLike = Any

########-------------------------------sub functions for the main training loop------------------------------#######

NUM_CLASSES=4

def initialize_model(key, indices, model):
    """
    Initializes the model
    -----------
    :param key: PRNG key
    :param indices: ndarray of valid pixel indices
    :param model: a `flax.module` instance
    -----------
    returns: variables['params'] and variables['batch_stats'], 
             initialized variables of the models
    """
    input_shape = (1, indices.shape[0], 1)
    key, dropout_key = jax.random.split(key)
    def init(*args):
        return model.init(*args)
    variables = jax.jit(init, backend='cpu')({'params': key, 'dropout': dropout_key}, 
                                             jnp.ones(input_shape))
    return variables['params'], variables['batch_stats'] 

def cross_entropy_loss(logits, labels):
    """
    Usual cross entropy loss function.
    ----------
    :param logits: logits output by the model
    :param labels: data labels
    ----------
    returns: cross entropy loss
    """
    one_hot_labels = common_utils.onehot(labels, num_classes=NUM_CLASSES)
    cross_entropy = optax.softmax_cross_entropy(logits, one_hot_labels)
    return jnp.mean(cross_entropy)

def compute_metrics(logits, labels):
    """
    Function to create training/testing metrics.
    ----------
    :param logits: logits output by the model
    :param labels: data labels
    ----------
    returns: metrics, a dict object.
             metrics['loss']=loss, metrics['accuracy']=accuracy
    """    
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {'loss': loss, 'accuracy': accuracy}
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    return metrics

def create_learning_rate_fn(config, base_learning_rate: float, steps_per_epoch: int):
    """
    Create learning rate scheduler function.
    Schedule is cosine decay with initial warmup.
    ----------
    :param config: training config file
    :param base_learning_rate: the learning rate after warmup epochs
    :param steps_per_epoch: gradient update steps per epoch
    ----------
    returns: schedule function.
    """
    warmup_fn = optax.linear_schedule(init_value=8e-5, 
                                      end_value=base_learning_rate,
                                      transition_steps=config['warmup_epochs'] * steps_per_epoch)
    cosine_epochs = max(config['num_epochs'] - config['warmup_epochs'], 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                            decay_steps=cosine_epochs * steps_per_epoch)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, cosine_fn],
                                       boundaries=[config['warmup_epochs'] * steps_per_epoch])
    return schedule_fn

def train_step(state, batch, learning_rate_fn):
    """
    Perform a single training step.
    ----------
    :param state: a `train_state.TrainState` instance
    :param batch: a batch of the training data
    :param learning_rate_fn: learning rate scheduling function
    ----------
    returns: new_state: updated `train_state.TrainState` instance
             metrics: training metrics as a dict
    """
    init_rng = jax.random.PRNGKey(2)
    dropout_rng = jax.random.fold_in(init_rng, state.step)
    
    def loss_fn(params):
        """
        Loss function used for training.
        """
        logits, new_model_state = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                                 rngs={'dropout': dropout_rng},
                                                 inputs=batch['maps'],
                                                 mutable=['batch_stats'])
        
        loss = cross_entropy_loss(logits, batch['labels'])
        weight_penalty_params = jax.tree_leaves(params)
        weight_decay = 0.0002
        weight_l2 = sum([jnp.sum(x ** 2)                   #l_2 decay for kernels
                        for x in weight_penalty_params
                        if x.ndim > 1])
        
        weight_penalty = weight_decay * 0.5 * weight_l2
        loss = loss + weight_penalty
        
        return loss, (new_model_state, logits)

    step = state.step
    dynamic_scale = state.dynamic_scale
    lr = learning_rate_fn(step)

    if dynamic_scale:
        grad_fn = dynamic_scale.value_and_grad(loss_fn, 
                                               has_aux=True, 
                                               axis_name='batch')
        dynamic_scale, is_fin, aux, grads = grad_fn(state.params)
        # dynamic loss takes care of averaging gradients across replicas
    else:
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        aux, grads = grad_fn(state.params)
        # Re-use same axis_name as in the call to `pmap(...train_step...)` below.
        grads = jax.lax.pmean(grads, axis_name='batch')
    
    new_model_state, logits = aux[1]
    metrics = compute_metrics(logits, batch['labels'])
    metrics['learning_rate'] = lr

    new_state = state.apply_gradients(grads=grads, 
                                      batch_stats=new_model_state['batch_stats'])
    
    if dynamic_scale:
        # if is_fin == False the gradients contain Inf/NaNs and optimizer state and
        # params should be restored (= skip this step).
        new_state = new_state.replace(opt_state=jax.tree_map(functools.partial(jnp.where, is_fin),
                                                             new_state.opt_state,
                                                             state.opt_state),
                                      params=jax.tree_map(functools.partial(jnp.where, is_fin),
                                                          new_state.params,
                                                          state.params))
        metrics['scale'] = dynamic_scale.scale

    return new_state, metrics

def test_step(state, batch):
    """
    Perform testing step.
    ----------
    :param state: a `train_state.TrainState` instance
    :param batch: a batch of the training data
    ----------
    returns: testing metrics, as a dict
    """
    variables = {'params': state.params, 'batch_stats': state.batch_stats}
    logits = state.apply_fn(variables, batch['maps'], train=False, mutable=False)
    return compute_metrics(logits, batch['labels'])

def prepare_tf_data(xs):
    """
    Convert an input batch coming from a `tf.data.Dataset` object to numpy arrays.
    ----------
    :param xs: input data tensors
    ----------
    returns: per device batched ndarrays converted from input `tf.tensor` objects
    """
    local_device_count = jax.local_device_count()
    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy() 
        # reshape (host_batch_size, num_pixels, 1) to
        # (local_devices, device_batch_size, num_pixels, 1)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_map(_prepare, xs)

def create_input_iter(dataset):
    """
    Create iterables from `tf.data.Datasets` and prefetch to device.
    ----------
    :param dataset: dataset container class, see `input_pipeline.py`
    ----------
    returns: an iterable object containing the data
    """
    it = map(prepare_tf_data, dataset)
    it = flax.jax_utils.prefetch_to_device(it, 2)
    return it

class TrainState(train_state.TrainState):
    """
    Simple addition to the generic `train_state.TrainState` for dynamic_scale support.
    """
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale

def restore_checkpoint(state, workdir):
    """
    Helper function to restore the module state from a saved checkpoint.
    ---------
    :param state: a `train_state.TrainState` instance
    :param workdir: string: path to saved checkpoints.
    ---------
    returns: module state with checkpoint restored.
    """
    return checkpoints.restore_checkpoint(workdir, state)

def save_checkpoint(state, workdir):
    """
    Helper function to save the current module state in a checkpoint.
    ---------
    :param state: a `train_state.TrainState` instance
    :param workdir: string: path to saved checkpoints.
    ---------
    returns: a saved checkpoint.
    """
    if jax.process_index() == 0:
    # get train state from the first replica
        state = jax.device_get(jax.tree_map(lambda x: x[0], state))
        step = int(state.step)
        checkpoints.save_checkpoint(workdir, state, step, keep=5, overwrite=True)
        
cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, 'x'), 'x')

def sync_batch_stats(state):
    """
    Sync the `nn.BatchNorm` module statistics across replicas (devices).
    ----------
    :param state: a `train_state.TrainState` instance
    ----------
    returns: `nn.BatchNorm` batch stats replaced onto the new `train_state.TrainState` instance.
    """
    # Each device has its own version of the running average batch norm statistics and
    # we sync them before evaluation.
    return state.replace(batch_stats=cross_replica_mean(state.batch_stats))

def create_train_state(rng, config,
                       model, indices, nside, learning_rate_fn):
    """
    Create initial training state.
    -----------
    :param rng: PRNG used to initialize the model. This will be split into two,
                one for model parameters, one for the dropout layers.
    :param config: hyperparameter config file
    :param model: Model instance
    :param indices: ndarray of valid map indices
    :param nside: nside of the input maps
    :param learning_rate_fn: callable learning rate scheduling function
    -----------
    returns: state, `flax.train_state.TrainState` object. 
    """
    dynamic_scale = None
    platform = jax.local_devices()[0].platform
    
    if config['half_precision'] and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None

    params, batch_stats = initialize_model(rng, indices, model)
    
    tx = optax.sgd(learning_rate=learning_rate_fn,
                   momentum=config['momentum'],
                   nesterov=config['nesterov'])
    
    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats,
                              dynamic_scale=dynamic_scale)
    return state

#########------------------------------------main training loop----------------------------------------#########

def train_and_test(dataset, train_data, test_data, model, config) -> TrainState:
    """
    Execute model training and evaluation loop.
    ------------
    :param dataset: dataset container class
    :param train_data: a `tf.data.Dataset`
    :param test_data: a `tf.data.Dataset`
    :param model: model to be trained, `flax.module` instance
    :param config: hyperparameter configuration dictionary for training and evaluation.
    ------------
    returns: Final TrainState, `flax.train_state.TrainState` object.
    """
    workdir = config['workdir']
    indices = model.indices
    nside = model.nside
    
    writer = metric_writers.create_default_writer(logdir=workdir, 
                                                  just_logging=jax.process_index() != 0)

    rng = jax.random.PRNGKey(0)
    rng, init_rng = jax.random.split(rng)
    dropout_rngs = flax.jax_utils.replicate(rng)

    if config['batch_size'] % jax.device_count() > 0:
        raise ValueError('Batch size must be divisible by the number of devices')
    
    local_batch_size = config['batch_size'] // jax.process_count()

    platform = jax.local_devices()[0].platform

    if config['half_precision']:
        if platform == 'tpu':
            input_dtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32
    
    train_iter = create_input_iter(dataset=train_data)
    
    test_iter = create_input_iter(dataset=test_data)

    steps_per_epoch = dataset.train_size // config['batch_size']
    
    num_steps = int(steps_per_epoch * config['num_epochs'])
    
    steps_per_test = dataset.test_size // config['batch_size']

    steps_per_checkpoint = steps_per_epoch

    base_learning_rate = config['init_lr'] * config['batch_size'] / 256.
    
    model = model

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate, steps_per_epoch)

    state = create_train_state(init_rng, config, model, indices, nside, learning_rate_fn)
    
    #state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    # step_offset = int(state.step)
    step_offset = 0
    
    state = flax.jax_utils.replicate(state)
    
    #partial wrapper is there because `learning_rate_fn` is not replicated.
    p_train_step = jax.pmap(functools.partial(train_step, learning_rate_fn=learning_rate_fn),
                            axis_name='batch', donate_argnums=(0,))
    
    p_test_step = jax.pmap(test_step, axis_name='batch')

    train_metrics = []
    hooks = []
    if jax.process_index() == 0:
        hooks += [periodic_actions.Profile(num_profile_steps=10, logdir=workdir)]
    train_metrics_last_t = time.time()
    logging.info('Starting initial compilation...')
    
    pbar = trange(step_offset, num_steps)
    for step, batch in zip(pbar, train_iter):
        state, metrics = p_train_step(state, batch)
        
        for h in hooks:
            h(step)
        if step == step_offset:
            logging.info('Initial compilation completed.')
            
        epoch = step // steps_per_epoch
        train_metrics.append(metrics)
        train_metrics_ = common_utils.get_metrics(train_metrics)
        train_metrics_ = {f'train_{k}': v for k, v in jax.tree_map(lambda x: x.mean(), 
                                                                  train_metrics_).items()}
        pbar.set_description(f"Epoch {epoch+1}/{config['num_epochs']}", refresh=True)
        pbar.set_postfix({'train_loss': train_metrics_['train_loss'],
                          'learning_rate': train_metrics_['train_learning_rate']}, refresh=True)
        
        if config['log_every_step']:
            if (step + 1) % config['log_every_step'] == 0:
                summary = {f'train_{k}': v for k, v in jax.tree_map(lambda x: x.mean(), 
                                                                    train_metrics_).items()}
                summary['steps_per_second'] = config['log_every_step'] / (time.time() - train_metrics_last_t)
                writer.write_scalars(step + 1, summary)
                train_metrics = []
                train_metrics_last_t = time.time()

        if (step + 1) % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            test_metrics = []

            # sync batch statistics across replicas
            state = sync_batch_stats(state)
            for _ in trange(steps_per_test):
                test_batch = next(test_iter)
                metrics = p_test_step(state, test_batch)
                test_metrics.append(metrics)
            
            test_metrics = common_utils.get_metrics(test_metrics)
            summary = jax.tree_map(lambda x: x.mean(), test_metrics)
            logging.info('test epoch: %d, test_loss: %.4f, test_accuracy: %.2f',
                         epoch+1, summary['loss'], summary['accuracy'] * 100)
            writer.write_scalars(step + 1, 
                                 {f'test_{key}': val for key, val in summary.items()})
            writer.flush()
                
        if (step + 1) % steps_per_checkpoint == 0 or step + 1 == num_steps:
            state = sync_batch_stats(state)
            save_checkpoint(state, workdir)

    # Wait until computations are done before exiting
    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()

    return state



