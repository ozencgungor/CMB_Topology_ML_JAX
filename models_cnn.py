import numpy as np
import jax.numpy as jnp
import flax.linen as nn

import functools
from functools import partial
from typing import Any, Callable, Sequence, Tuple

from absl import logging

from src import nngcn

ModuleDef = Any
Array = Any

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
        inputs['maps']
        return inputs
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
        inputs['maps'] = x
        return inputs
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
    x = jnp.concatenate(x, axis=axis)
    xs[0]['maps'] = x
    return xs[0]

def add(*xs):
    """
    Simple addition function that adds inputs while passing through xs['nside'] and xs['indices'].
    Assumes all inputs have the same 'nside' and 'indices' (addition wouldn't be possible otherwise anyway).
    ---------
    args:
    xs: inputs of the form {'nside': nside, 'indices': indices, 'maps': array like}
    ---------
    returns: {'nside': nside, 'indices': indices, 'maps': x1['maps']+x2['maps']+...} 
             for x1, x2, ... in xs
    """
    x = *(elem['maps'] for elem in xs),
    x = sum(x)
    xs[0]['maps'] = x
    return xs[0]

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
    x['maps'] = xout
    return x 
    
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
    def __call__(self, inputs):
        Conv = nngcn.HealpyChebyshevConv_v2
        PseudoConv = nngcn.HealpyPseudoConv
        
        x1 = Conv(K=4, Fout=self.filters, n_neighbors=8, p=0)(inputs)
        x1 = self.act(x1)
        x1 = self.norm(x1)
        x1 = Conv(K=8, Fout=self.filters, n_neighbors=8, p=0)(x1)
        x1 = self.act(x1)
        x1 = self.norm(x1)
        x1 = Conv(K=2, Fout=self.filters, n_neighbors=8, p=1)(x1)
        x1 = self.act(x1)
        x1 = self.norm(x1)

        x2 = Conv(K=8, Fout=self.filters, n_neighbors=8, p=0)(inputs)
        x2 = self.act(x2)
        x2 = self.norm(x2)
        x2 = Conv(K=2, Fout=self.filters, n_neighbors=8, p=1)(x2)
        x2 = self.act(x2)
        x2 = self.norm(x2)
        
        x3 = Conv(K=12, Fout=2*self.filters, n_neighbors=20, p=0)(inputs)
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
    def __call__(self, inputs):
        SepConv = nngcn.HealpySeparableChebyshevConv      
        
        res = inputs
        x = SepConv(K=6, Fout=self.filters, depth_multiplier=self.depth_mul, n_neighbors=8)(inputs)
        x = self.act(x)
        x = self.norm(x)
        x = SepConv(K=10, Fout=self.filters, depth_multiplier=self.depth_mul, n_neighbors=20)(x)
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
    def __call__(self, inputs):
        Conv = nngcn.HealpyChebyshevConv_v2        
        
        res = inputs
        x = Conv(K=6, Fout=self.filters, n_neighbors=8, p=0)(inputs)
        x = self.act(x)
        x = self.norm(x)
        x = Conv(K=10, Fout=self.filters, n_neighbors=20, p=0)(x)
        x = self.act(x)
        x = self.norm(x)   
        x = add(x, res)
        return x
    
class MiddleBlock(nn.Module):
    """
    Transition block between residual blocks.
    -----------
    :param K: order of polynomials to use.
    :param filters: number of features output by the block.
    :param norm: callable norm module from nn.module
    :param act: callable activation function
    """
    K: int
    filters: int
    norm: ModuleDef
    act: ModuleDef
    
    @nn.compact
    def __call__(self, inputs):
        Conv = nngcn.HealpyChebyshevConv_v2
        PseudoConv = nngcn.HealpyPseudoConv       
        
        x = Conv(K=self.K, Fout=self.filters, n_neighbors=8)(inputs)
        x = self.act(x)
        x = self.norm(x)
        x = PseudoConv(p=1, Fout=self.filters)(x)
        x = self.act(x)
        x = self.norm(x)
        return x
    
class FinalBlock(nn.Module):
    """
    Final block before top.
    :param filters: number of features output by the block.
    :param norm: callable norm module from nn.module
    :param act: callable activation function
    """
    filters: int
    norm: ModuleDef
    act: ModuleDef
    
    @nn.compact
    def __call__(self, inputs):
        Conv = nngcn.HealpyChebyshevConv_v2
        
        x = Conv(K=8, Fout=self.filters, n_neighbors=8, p=0)(inputs)
        x = self.act(x)
        x = self.norm(x)
        x = Conv(K=12, Fout=self.filters, n_neighbors=20, p=0)(x)
        x = self.act(x)
        x = self.norm(x)
        return x
        
class HealpyCNN(nn.Module): 
    """
    Graph Convolutional Network for healpy maps.
    -----------
    :param nside: int, nside of the input maps
    :param indices: ndarray, valid(unmasked) indices of the input maps
    :param include_top: bool, whether to include the final GAP and dense classifier modules
    :param num_classes: number of classes to classify from
    :param res_depth: number of  (2*convolutional layers) in residual blocks
    -----------
    returns: if include_top=True will return logits, 
             otherwise will return {'nside': final_nside, 'indices': final_indices, 'maps': maps}
    """
    nside: int
    indices: Array
    dropout_rate: float = 0.1
    include_top: bool = True
    num_classes: int=3
    res_depth: int=3
    
    @nn.compact
    def __call__(self, inputs, train: bool = True):
        relu = _non_hp_func(nn.relu)
        BN = _non_hp_module(partial(nn.BatchNorm, use_running_average=not train,
                                    momentum=0.9, epsilon=1e-5))
        LN = _non_hp_module(partial(nn.LayerNorm, use_bias=True, use_scale=True))
        Dropout = _non_hp_module(partial(nn.Dropout, rate=self.dropout_rate, deterministic=not train))        
        
        nside = self.nside
        indices = self.indices
        
        x = {'nside': nside, 'indices': indices, 'maps': inputs}
        x = WideBlock(16, norm=LN, act=relu, name='init_block_1')(x)
        x = Dropout(x)
        x = WideBlock(32, norm=LN, act=relu, name='init_block_2')(x)
        x = Dropout(x)
        
        #prints are for intermediate shape checks.
        for i in range(self.res_depth):
            x = ResidualConvBlock(128, norm=BN, act=relu)(x)
            x = Dropout(x)
        #print('nside: ', x['nside'], 'intermediate shape: ', x['maps'].shape)
        
        x = MiddleBlock(8, 256, norm=LN, act=relu)(x)
        x = Dropout(x)
        #print('nside: ', x['nside'], 'intermediate shape: ', x['maps'].shape)
        
        for i in range(self.res_depth):
            x = ResidualConvBlock(256, norm=BN, act=relu)(x)
            x = Dropout(x)
        #print('nside: ', x['nside'], 'intermediate shape: ', x['maps'].shape)
        
        x = MiddleBlock(8, 512, norm=LN, act=relu)(x)
        x = Dropout(x)
        #print('nside: ', x['nside'], 'intermediate shape: ', x['maps'].shape)
        
        for i in range(self.res_depth):
            x = ResidualConvBlock(512, norm=BN, act=relu)(x)
            x = Dropout(x)
        #print('nside: ', x['nside'], 'intermediate shape: ', x['maps'].shape)
        x = FinalBlock(512, norm=BN, act=relu)(x)
        
        if self.include_top:
            x = jnp.mean(x['maps'], axis=-2) #global_avg_pool
            x = nn.Dense(self.num_classes)(x)
        return x
        
        
        
        
        
        