import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn

import functools
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from absl import logging

from src import nngcn

ModuleDef = Any
Array = Any

def _non_hp_module_v2(module):
    """
    Decorator to have linen modules ignore inputs['nside'] and inputs['indices'].
    Mainly for use with usual linen layers like Dense, Conv, BatchNorm, LayerNorm etc.
    ---------
    :param module: a linen module with a __call__() method.
    ---------
    returns a module that will only act on x['maps'] when called on x.
    i.e module(*args, **kwargs)({'a': a, 'b': b, 'maps': maps})
           = {'a': a, 'b': b, 'maps': module(*agrs, **kwargs)({'a': a, 'b': b, 'maps': maps})}
           
    usage looks like `Dense = _non_hp_module(nn.Dense)`
                     `y = Dense(*args, *kwargs)(x)`
                     `BatchNorm = _non_hp_module(partial(nn.BatchNorm, use_running_average=not train,
                                                         momentum=0.9, epsilon=1e-5))`
                     `y = BatchNorm(x)`  etc.
    """
    @functools.wraps(module)
    def non_hp_decorator(inputs, *args, **kwargs):
        x = inputs['maps']
        mod = functools.partial(module, *args, **kwargs)
        x = mod()(x)
        output = {'nside': inputs['nside'], 'indices': inputs['indices'], 'maps': x}
        return output
    return non_hp_decorator

def _non_hp_module(module):
    """
    Decorator to have linen modules ignore inputs['nside'] and inputs['indices'].
    Mainly for use with usual linen layers like Dense, Conv, BatchNorm, LayerNorm etc.
    ---------
    :param module: a linen module with a __call__() method.
    ---------
    returns a module that will only act on x['maps'] when called on x.
    i.e module(*args, **kwargs)({'a': a, 'b': b, 'maps': maps})
           = {'a': a, 'b': b, 'maps': module(*agrs, **kwargs)({'a': a, 'b': b, 'maps': maps})}
           
    usage looks like `Dense = _non_hp_module(nn.Dense)`
                     `y = Dense(x, features=...)`
                     `BatchNorm = _non_hp_module(partial(nn.BatchNorm, use_running_average=not train,
                                                         momentum=0.9, epsilon=1e-5))`
                     `y = BatchNorm(x)`  etc.
    """
    @functools.wraps(module)
    def non_hp_decorator(inputs, *args, **kwargs):
        x = inputs['maps']
        mod = module(*args, **kwargs)
        x = mod(x)
        output = {key: value for (key, value) in inputs.items()}
        output['maps'] = x
        return output
    return non_hp_decorator

def _non_hp_func(func):
    """
    Decorator to have functions ignore inputs['nside'] and inputs['indices'].
    Use this for activation functions.
    --------
    :param func: callable function to decorate
    --------
    returns a module that will only act on x['maps'] when called on x.
    i.e f({'a': a, 'b': b, 'maps': maps}, 
          *args, **kwargs) = {'a': a, 'b': b, 'maps': f(maps, *agrs, **kwargs)}
          
    usage looks like `relu = _non_hp_func(nn.relu)`
                     `y = relu(x)
              
    """
    @functools.wraps(func)
    def non_hp_decorator(inputs, *args, **kwargs):
        x = inputs['maps']
        x = func(x, *args, **kwargs)
        output = {key: value for (key, value) in inputs.items()}
        output['maps'] = x
        return output
    return non_hp_decorator

def concatenate(*xs, axis=-1):
    """
    Simple concatenate function that concatenates inputs while passing through xs['nside'] and xs['indices'].
    Assumes all inputs have the same 'nside' and 'indices' (concatenation wouldn't be possible otherwise anyway).
    ---------
    args:
    xs: inputs of the form {'nside': nside, 'indices': indices, 'maps': array like}
    ---------
    returns: {'nside': nside, 'indices': indices, 'maps': jnp.concatenate((x1['maps'], x2['maps'], ...), axis=axis)}
             for x1, x2, ... in xs
    """
    x = *(elem['maps'] for elem in xs),
    xout = jnp.concatenate(x, axis=axis)
    output = {key: value for (key, value) in xs[0].items()}
    output['maps'] = xout
    return output

def add(*xs):
    """
    Simple addition function that adds inputs while passing through xs['nside'] and xs['indices'].
    Assumes all inputs have the same 'nside' and 'indices' (addition wouldn't be possible otherwise anyway).
    ---------
    args:
    xs: inputs of the form {'nside': nside, 'indices': indices, 'maps': array like}
    ---------
    returns: {'nside': nside, 'indices': indices, 'maps': x1['maps']+x2['maps'+...} 
             for x1, x2, ... in xs
    """
    x = *(elem['maps'] for elem in xs),
    xout = sum(x)
    output = {key: value for (key, value) in xs[0].items()}
    output['maps'] = xout    
    return output  

def transpose(x, axes):
    """
    Simple addition function that transposes the maps with the given 
    indices while passing through xs['nside'] and xs['indices'].
    ---------
    args:
    x: input of the form {'nside': nside, 'indices': indices, 'maps': array like}
    ---------
    returns: {'nside': nside, 'indices': indices, 'maps': jnp.transpose(x, axes=axes)}
    """
    maps = x['maps']
    xout = jnp.transpose(maps, axes=axes)
    output = {key: value for (key, value) in x.items()}
    output['maps'] = xout
    return output

class WideBlock(nn.Module):
    """
    Initial blocks.
    ---------
    :param filters: number of features output by the block. The block will output 4*filters many features.
    :param norm: callable norm module from nn.module
    :param act: callable activation function
    ---------
    """
    filters: int
    norm: ModuleDef
    act: ModuleDef    
    
    @nn.compact
    def __call__(self, x):
        Conv = nngcn.HealpyChebyshevConv_v2
        PseudoConv = nngcn.HealpyPseudoConv
        
        x1 = Conv(K=4, Fout=self.filters, n_neighbors=8, p=0)(x)
        x1 = self.act(x1)
        x1 = self.norm(x1)
        x1 = Conv(K=2, Fout=self.filters, n_neighbors=8, p=1)(x1)
        x1 = self.act(x1)
        x1 = self.norm(x1)

        x2 = Conv(K=6, Fout=self.filters, n_neighbors=8, p=0)(x)
        x2 = self.act(x2)
        x2 = self.norm(x2)
        x2 = Conv(K=2, Fout=self.filters, n_neighbors=8, p=1)(x2)
        x2 = self.act(x2)
        x2 = self.norm(x2)
        
        x3 = Conv(K=8, Fout=2*self.filters, n_neighbors=20, p=0)(x)
        x3 = self.act(x3)
        x3 = self.norm(x3)
        x3 = Conv(K=2, Fout=2*self.filters, n_neighbors=8, p=1)(x3)
        x3 = self.act(x3)
        x3 = self.norm(x3)
        
        x = concatenate(x1, x2, x3, axis=-1)
        return x
    
class ResidualSeparableBlock(nn.Module):
    """
    Separable residual blocks.
    ---------
    :param filters: number of features output by the block.
    :param depth_mul: depth multiplier for separable convolutions
    :param norm: callable norm module from nn.module
    :param act: callable activation function
    ---------
    """
    filters: int
    depth_mul: int
    norm: ModuleDef
    act: ModuleDef
    
    @nn.compact
    def __call__(self, x):
        SepConv = nngcn.HealpySeparableChebyshevConv
        
        res = x
        x = SepConv(K=4, Fout=self.filters, depth_multiplier=self.depth_mul, n_neighbors=8)(x)
        x = self.act(x)
        x = self.norm(x)
        x = SepConv(K=8, Fout=self.filters, depth_multiplier=self.depth_mul, n_neighbors=8)(x)
        x = self.act(x)
        x = self.norm(x)  
        x = add(x, res)
        return x
        
class ResidualConvBlock(nn.Module):
    """
    Convolutional residual blocks.
    ---------
    :param filters: number of features output by the block.
    :param norm: callable norm module from nn.module
    :param act: callable activation function
    ---------
    """
    filters: int
    norm: ModuleDef
    act: ModuleDef
    
    @nn.compact
    def __call__(self, x):
        Conv = nngcn.HealpyChebyshevConv_v2
        
        res = x
        x = Conv(K=6, Fout=self.filters, n_neighbors=8, p=0)(x)
        x = self.act(x)
        x = self.norm(x)
        x = Conv(K=10, Fout=self.filters, n_neighbors=20, p=0)(x)
        x = self.act(x)
        x = self.norm(x)   
        x = add(x, res)
        return x

class DenseBlock(nn.Module):
    """
    Dense Block.
    ---------
    :param mlp_dim:
    ---------
    """
    mlp_dim: int
    
    @nn.compact
    def __call__(self, x):
        gelu = _non_hp_func(nn.gelu)
        Dense = _non_hp_module(nn.Dense)
        print(x['maps'].shape, "shape at entrance to dense_block")
        
        y = Dense(x, self.mlp_dim, 
                  dtype=jnp.float32,
                  kernel_init=nn.initializers.xavier_uniform(),
                  bias_init=nn.initializers.normal(stddev=1e-6))
        y = gelu(y)
        y = Dense(y, x['maps'].shape[-1], 
                  dtype=jnp.float32,
                  kernel_init=nn.initializers.xavier_uniform(),
                  bias_init=nn.initializers.normal(stddev=1e-6))
        print(y['maps'].shape, "shape after denseblock")
        return y
    
class MixerBlock(nn.Module):
    """
    Mixer block module.
    -----------
    :param tokens_mlp_dim:
    :param channels_mlp_dim:
    -----------
    """
    tokens_mlp_dim: int
    channels_mlp_dim: int
    
    @nn.compact
    def __call__(self, x):
        LayerNorm = _non_hp_module(nn.LayerNorm)
        relu = _non_hp_func(nn.relu)
        gelu = _non_hp_func(nn.gelu)
        Conv = nngcn.HealpyChebyshevConv_v2
        
        y = LayerNorm(x)
        print(y['maps'].shape, "shape at entrance to mixerblock")
        #we can add in a cheby conv here if it can fit in memory ofc
        y = Conv(K=6, Fout=y['maps'].shape[-1], n_neighbors=8, p=0)(y)
        y = gelu(y)
        y = LayerNorm(y)
        y = transpose(y, axes=(0, 2, 1))
        print(y['maps'].shape, "shape after transposition in mixerblock before denseblock")
        y = DenseBlock(self.tokens_mlp_dim, name='token_mixing')(y)
        print(y['maps'].shape, "shape after 1st denseblock")
        y = transpose(y, axes=(0, 2, 1))
        print(y['maps'].shape, "shape after transposition in mixerblock after denseblock")
        x = add(x, y)
        y = LayerNorm(x)
        #also here
        y = Conv(K=6, Fout=y['maps'].shape[-1], n_neighbors=8, p=0)(y)
        y = gelu(y)
        y = LayerNorm(y)        
        y = DenseBlock(self.channels_mlp_dim, name='channel_mixing')(y)
        print(y['maps'].shape, "shape after 2nd denseblock in mixerblock before addition")
        return add(x, y)  


class Transformer(nn.Module):
    """
    Graph MLP-Mixer/Transformer for healpy maps
    -----------
    :param nside: int, nside of the input maps
    :param indices: ndarray, valid(unmasked) indices of the input maps
    :param nside_super: nside of the superpixel map. 
                        will create 12*nside_super**2 patches from the input maps
    :param K_p: order of the chebyshev polynomial to use in the patch creation layer
    :param conv_depth: number of residual conv blocks before the mixer
    :param conv_features: features output by the pre mixer cnn 
    :param num_mixer_blocks: number of mlp mixer blocks
    :param hidden_dim: number of features output by the patcher layer
    :param tokens_mlp_dim:
    :param channels_mlp_dim:
    :param num_classes: number of classes to classify from
    :param include_top: bool, whether to include the final GAP and dense classifier modules
    -----------
    returns: if include_top=True will return logits, 
             otherwise will return {'nside': final_nside, 'indices': final_indices, 'maps': maps}    
    """
    nside: int
    indices: Array
    nside_super: int
    K_p: int
    conv_depth: int
    conv_features: int
    hidden_dim: int
    num_mixer_blocks: int
    tokens_mlp_dim: int
    channels_mlp_dim: int
    num_classes: int
    superpix_threshold: float = 0.5
    include_top: bool = True
        
    @nn.compact
    def __call__(self, inputs, train: bool = True):
        relu = _non_hp_func(nn.relu)
        GraphTransform = nngcn.HealpyGraphTransformer
        LayerNorm = _non_hp_module(nn.LayerNorm)
        BatchNorm = _non_hp_module(partial(nn.BatchNorm, use_running_average=not train,
                                   momentum=0.9, epsilon=1e-5))        
        x = {'nside': self.nside, 'indices': self.indices, 'maps': inputs}
        
        x = WideBlock(self.conv_features//4, norm=BatchNorm, act=relu, name='init_conv_1')(x)
        for i in range(self.conv_depth):
            x = ResidualConvBlock(self.conv_features, norm=BatchNorm, act=relu)(x)
       
        x = GraphTransform(nside_sup=self.nside_super, 
                           K_p=self.K_p, 
                           Fout=self.hidden_dim, 
                           threshold=self.superpix_threshold,
                           n_neighbors_p=20, 
                           einsum=False)(x) #initial conv to create embeddings.
        print(x['maps'].shape, "shape after transformer")
        
        for i in range(self.num_mixer_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        
        if self.include_top:
            x = LayerNorm(x)
            out = jnp.mean(x['maps'], axis=1) #global_avg_pool
            out = nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros)(out)
        
        return out
        
                           
        
    
    
    







