import numpy as np
import jax.numpy as jnp
import flax.linen as nn

import functools
from functools import partial
from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

from src import nngcn

ModuleDef = Any
Array = Any
PRNGKey = Any
Shape = Tuple[int]
Dtype = Any

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

#hi-res feature extraction CNN modules:
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

        x2 = Conv(K=8, Fout=self.filters, n_neighbors=8, p=0)(x)
        x2 = self.act(x2)
        x2 = self.norm(x2)
        x2 = Conv(K=2, Fout=self.filters, n_neighbors=8, p=1)(x2)
        x2 = self.act(x2)
        x2 = self.norm(x2)
        
        x3 = Conv(K=12, Fout=2*self.filters, n_neighbors=20, p=0)(x)
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
        x = SepConv(K=6, Fout=self.filters, depth_multiplier=self.depth_mul, n_neighbors=8)(x)
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

class AddPositionEmbs(nn.Module):
    """
    Adds learnable positional embeddings to the inputs.
    -------------
    :param posemb_init: positional embedding initializer.
    -------------
    """
    posemb_init: Callable[[PRNGKey, Shape, Dtype], Array]

    @nn.compact
    def __call__(self, inputs):
        """
        Applies the AddPositionEmbs module.
        ----------------
        :param inputs: Inputs to the layer 
                       in dict format {'nside': int, 'indices': Array, 'maps': Array}
        ----------------
        Returns:
        Output tensor with shape `(batch, sequence, in_dim)`.
        """
            # inputs.shape is (batch_size, seq_len, emb_dim).
        assert inputs['maps'].ndim == 3, ('Number of dimensions should be 3,'
                                          ' but it is: %d' % inputs.ndim)
        pos_emb_shape = (1, inputs['maps'].shape[-2], inputs['maps'].shape[-1])
        posemb = self.param('pos_embedding', self.posemb_init, pos_emb_shape)
        inputs['maps'] = inputs['maps'] + posemb
        return inputs
    
class MlpBlock(nn.Module):
    """
    Transformer MLP / feed-forward block.
    -----------
    :param mlp_dim: features output by the hidden module
    :param out_dim: features output by the last module
    :param dropout_rate: dropout rate
    :param kernel_init: kernel initializer for the fc modules.
    :param bias_init: bias initializer for the fc modules.
    :param dtype: dtype of the inputs
    -----------
    """

    mlp_dim: int
    out_dim: Optional[int] = None
    dropout_rate: float = 0.1
    kernel_init: Callable[[PRNGKey, Shape, Dtype],
                        Array] = nn.initializers.xavier_uniform()
    bias_init: Callable[[PRNGKey, Shape, Dtype],
                      Array] = nn.initializers.normal(stddev=1e-6)
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """
        Applies MlpBlock module.
        ----------
        :param inputs: Inputs to the layer. 
                       In dict format {'nside': nside, 'indices': indices, 'maps': maps}
        :param deterministic: Dropout will not be applied when set to true.
        ----------
        returns: output after MlpBlock module in dict format as above
        """
        actual_out_dim = inputs['maps'].shape[-1] if self.out_dim is None else self.out_dim
        Dense = _non_hp_module(nn.Dense)
        gelu = _non_hp_func(nn.gelu)
        Dropout = _non_hp_module(partial(nn.Dropout, rate=self.dropout_rate, deterministic=deterministic))
        
        x = Dense(inputs, features=self.mlp_dim,
                  dtype=self.dtype,
                  kernel_init=self.kernel_init,
                  bias_init=self.bias_init)
        x = gelu(x)
        x = Dropout(x)
        x = Dense(x, features=actual_out_dim,
                  dtype=self.dtype,
                  kernel_init=self.kernel_init,
                  bias_init=self.bias_init)
        output = Dropout(x)
        return output
    
class GraphEncoderBlock(nn.Module):
    """
    Graph Transformer encoder layer.
    ------------
    :param inputs: input data.
    :param mlp_dim: dimension of the mlp on top of attention block.
    :param dtype: the dtype of the computation (default: float32).
    :param dropout_rate: dropout rate.
    :param attention_dropout_rate: dropout for attention heads.
    :param deterministic: bool, deterministic or not (to apply dropout).
    :param num_heads: Number of heads in nn.MultiHeadDotProductAttention
    ------------
    """

    mlp_dim: int
    num_heads: int
    dtype: Dtype = jnp.float32
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    class_token: bool = True

    @nn.compact
    def __call__(self, inputs, *, deterministic):
        """
        Applies GraphEncoderBlock module.
        ----------
        :param inputs: Inputs to the layer. 
                       In dict format {'nside': nside, 'indices': indices, 'maps': maps}
        :param deterministic: Dropout will not be applied when set to true.
        ----------
        returns: output after transformer encoder block in dict format as above
        """
        LayerNorm = _non_hp_module(functools.partial(nn.LayerNorm, dtype=self.dtype))
        Dropout = _non_hp_module(partial(nn.Dropout, rate=self.dropout_rate, deterministic=deterministic))        
        GraphMultiHeadAttention = nngcn.HealpyMultiHeadAttention_v2

        # Attention block.
        assert inputs['maps'].ndim == 3, f'Expected (batch, seq, hidden) got {inputs.shape}'
        x = LayerNorm(inputs)
        x = GraphMultiHeadAttention(num_heads=self.num_heads,
                                    class_token=self.class_token,
                                    dtype=self.dtype,
                                    kernel_init=nn.initializers.xavier_uniform(),
                                    broadcast_dropout=False,
                                    deterministic=deterministic,
                                    dropout_rate=self.attention_dropout_rate)(x)
        
        #also try just regular attention:
        #x = nn.MultiHeadDotProductAttention(dtype=self.dtype,
        #                                    kernel_init=nn.initializers.xavier_uniform(),
        #                                    broadcast_dropout=False,
        #                                    deterministic=deterministic,
        #                                    dropout_rate=self.attention_dropout_rate,
        #                                    num_heads=self.num_heads)(x, x)

        x = Dropout(x)
        x = add(x, inputs)

        # MLP block.
        y = LayerNorm(x)
        y = MlpBlock(mlp_dim=self.mlp_dim, 
                     dtype=self.dtype, 
                     dropout_rate=self.dropout_rate)(y, deterministic=deterministic)

        return add(x, y)
    
class GraphEncoder(nn.Module):
    """
    Graph Transformer Model Encoder.
    -----------
    :param num_blocks: number of layers
    :param num_heads: Number of heads in nn.MultiHeadDotProductAttention
    :param mlp_dim: dimension of the mlp on top of attention block
    :param dropout_rate: dropout rate.
    :param attention_dropout_rate: dropout rate in self attention.
    :param mask: attention mask
    """        

    num_blocks: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    add_position_embedding: bool = True
    class_token: bool = True
    dtype: Dtype = jnp.float32

    @nn.compact
    def __call__(self, x, *, train):
        """
        Applies Transformer model on the inputs.
        ---------
        :param x: Inputs to the layer.
                  In dict format {'nside': nside, 'indices': indices, 'maps': maps}
        :param train: Set to `True` when training.
        ---------
        returns: output of a transformer encoder in the format above.
        """
        assert x['maps'].ndim == 3  # (batch, len, emb)
        
        LayerNorm = _non_hp_module(functools.partial(nn.LayerNorm, dtype=self.dtype))
        Dropout = _non_hp_module(partial(nn.Dropout, rate=self.dropout_rate, deterministic = not train))  

        if self.add_position_embedding == True:
            x = AddPositionEmbs(posemb_init=nn.initializers.normal(stddev=0.02),  
                                name='posembed_input')(x)
            x = Dropout(x)

        # Input Encoder
        for block in range(self.num_blocks):
              x = GraphEncoderBlock(mlp_dim=self.mlp_dim,
                                    dropout_rate=self.dropout_rate,
                                    attention_dropout_rate=self.attention_dropout_rate,
                                    num_heads=self.num_heads,
                                    class_token=self.class_token)(x, deterministic=not train)
        encoded_output = LayerNorm(x, name='encoder_norm')
        return encoded_output
    
class GraphVisionTransformer(nn.Module):
    """
    Graph Vision Transformer with attention.
    --------------
    :param nside: int, nside of the input maps
    :param indices: ndarray, valid(unmasked) indices of the input maps
    :param nside_super: nside of the superpixel map. 
                        will create 12*nside_super**2 patches from the input maps
    :param K_p: order of the chebyshev polynomial to use in the patch creation layer
    :param conv_depth: number of residual conv blocks before the mixer
    :param conv_features: features output by the pre mixer cnn 
    :param hidden_dim: number of features output by the patcher layer
    :param positional_embedding: bool, whether to add positional embeddings
    :param num_encoder_blocks: number of mlp mixer blocks
    :param num_heads: number of attention heads. inputs['maps'].shape[-1] % num_heads == 0
    :param mlp_dim: 
    :param superpix_mask_threshold: optional float between 0 and 1. threshold for superpixel mask.
                                    if set, will apply attention mask to superpixels who have less 
                                    than 100*threshold% of their subpixels masked at nside before
                                    patch creation.
    :param dropout_rate:
    :param attention_dropout_rate:
    :param num_classes: number of classes to classify between
    :param include_top: bool, whether to include the final GAP and dense classifier modules
    --------------
    """
    nside: int
    indices: Array
    nside_super: int
    K_p: int
    conv_depth: int
    conv_features: int
    hidden_dim: int
    positional_embedding: bool
    num_encoder_blocks: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    classifier: str = 'token'
    num_classes: int = 4
    include_top: bool = True    
    superpix_mask_threshold: Optional[float] = 0   
    
    @nn.compact
    def __call__(self, inputs, train: bool = True):
        GraphTransform = nngcn.HealpyGraphTransformer
        relu = _non_hp_func(nn.relu)
        LayerNorm = _non_hp_module(nn.LayerNorm)
        BatchNorm = _non_hp_module(partial(nn.BatchNorm, use_running_average=not train,
                                   momentum=0.9, epsilon=1e-5))        
        x = {'nside': self.nside, 'indices': self.indices, 'maps': inputs}
        
        #initial small cnn for high-res feature extraction.
        x = WideBlock(self.conv_features//4, norm=BatchNorm, act=relu, name='init_conv_1')(x)
        for i in range(self.conv_depth):
            x = ResidualConvBlock(self.conv_features, norm=BatchNorm, act=relu)(x) 

        x = GraphTransform(nside_sup=self.nside_super, K_p=self.K_p, 
                           Fout=self.hidden_dim, n_neighbors_p=8, 
                           threshold=self.superpix_mask_threshold, einsum=False)(x)
        
        n, s, c = x['maps'].shape
        
        
        class_token = False
        if self.classifier == 'token':
            #we need to modify the adjacency matrix for class token to work.
            #the class token is added to the pos x[:,0,:], we can add a row and a column of `True` values
            #to the graph adjacency matrix to take that into account.
            cls = self.param('cls', nn.initializers.zeros, (1, 1, c))
            cls = jnp.tile(cls, [n, 1, 1])
            x['maps'] = jnp.concatenate([cls, x['maps']], axis=1)
            class_token = True
        print(x['maps'].shape, x['indices'].shape)
        
        x = GraphEncoder(num_blocks=self.num_encoder_blocks, num_heads=self.num_heads,
                         mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate,
                         attention_dropout_rate=self.attention_dropout_rate,
                         add_position_embedding=self.positional_embedding,
                         class_token=class_token)(x, train=train)
                 
        if self.include_top:
            if self.classifier == 'gap':  
                x = LayerNorm(x)
                out = jnp.mean(x['maps'], axis=-2) #global_avg_pool
                print(out.shape, 'shape before top with gap')
            elif self.classifier == 'token':
                x = x['maps']
                out = x[:,0]
                print(out.shape, 'shape before top with class token')
            out = nn.Dense(self.num_classes, kernel_init=nn.initializers.zeros)(out) 
        
        return out
                             
                             
            
       
            
            
       
