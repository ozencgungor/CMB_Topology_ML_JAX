import numpy as np

import scipy as scipy
from scipy import sparse
from scipy.sparse.linalg import eigsh

import healpy as hp

from tqdm import tqdm, trange

import time

from pygsp.graphs import SphereHealpix
from pygsp import filters

import jax.lax
import jax.numpy as jnp
from jax import grad, jit, vmap
import jax.random
from jax.test_util import check_grads
from jax.experimental import sparse as jaxsparse

import flax
import flax.linen as nn
from flax import jax_utils

import opt_einsum as oe

import functools
from functools import partial

from typing import (Any, Callable, Iterable, List, Optional, Sequence, Tuple,
                    Union)

PRNGKey = Any
Shape = Tuple[int, ...]
Dtype = Any
Array = Any
PrecisionLike = Union[None, str, jax.lax.Precision, Tuple[str, str],
                      Tuple[jax.lax.Precision, jax.lax.Precision]]

PaddingLike = Union[str, int, Sequence[Union[int, Tuple[int, int]]]]
LaxPadding = Union[str, Sequence[Tuple[int, int]]]

#TODO: finish wrapper on transformer and start testing it.

def get_L(nside, indices, n_neighbors):
    """
    Helper function to get the graph laplacian L for a healpix map.
    --------
    :param nside: nside of the map
    :param indices: (masked)indices of the map, 1D array. if the maps are masked, the index array should
                    only contain pixel indices for unmasked pixels.
    :param n_neighbors: number of neighbors for each pixel
    --------
    Returns: the graph laplacian L, jax BCOO array of shape (M, M)
    """
    if n_neighbors not in [8,20,40,80]:
        raise NotImplementedError(f"The requested number of neighbors {n_neighbors} is nor supported."
                                  f"Choose either 8, 20, 40 or 60.")
    
    def get_lmax(L):
        return 1.02 * eigsh(L, k=1, which='LM', return_eigenvectors=False)[0]
    
    def rescale_L(L, lmax=2, scale=1):
        """Rescale the Laplacian eigenvalues in [-scale,scale]."""
        L *= 2 * scale / lmax
        L -= sparse.identity(L.shape[0], format='csr', dtype=L.dtype)
        return L
    
    L = SphereHealpix(subdivisions=nside, indexes=indices, nest=True, 
                      k=n_neighbors, lap_type='normalized').L.astype(np.float32)
    L = sparse.csr_matrix(L)
    L = rescale_L(L, lmax=get_lmax(L), scale=0.75)
    L = L.tocoo()
    L = jaxsparse.BCOO((L.data, jnp.column_stack((L.row, L.col))), shape=L.shape)
    L = L.sort_indices()
    return L

def get_A(nside, indices, n_neighbors):
    """
    Helper function to get the graph adjacency matrix A for a healpix map.
    --------
    :param nside: nside of the map
    :param indices: (masked)indices of the map, 1D array. if the maps are masked, the index array should
                    only contain pixel indices for unmasked pixels.
    :param n_neighbors: number of neighbors for each pixel
    --------
    Returns: the graph adjacency matrix A, jax BCOO array of shape (M, M)
    """    
    
    A = SphereHealpix(subdivisions=nside, indexes=indices, nest=True, 
                      k=n_neighbors, lap_type='normalized').A.astype(np.int32)
    
    A = sparse.csr_matrix(A)
    A = A.tocoo()
    A = jaxsparse.BCOO((A.data, jnp.column_stack((A.row, A.col))), shape=A.shape)
    A = A.sort_indices()
    return A

def block_diag(L, block_size):
    """
    Turns M into block diagonal format where each block is of shape (block_size, block_size)
    by putting zeros in places that do not belong to a block
    ----------
    M: jax BCOO array
    block_size: size of blocks
    """
    data = L.data
    idx = L.indices
    row = idx[:,0]
    col = idx[:,1]
    block_num = row//block_size
    blocks_delim_low = block_num*block_size 
    blocks_delim_high = block_num*block_size + block_size
    data = (blocks_delim_low<=(col))*((col)<blocks_delim_high)*data
    new_L = jaxsparse.BCOO((data, idx), shape=L.shape)
    return new_L

@jax.jit
def index_mapping_fun(a, b):
    """
    a: COO indices (jaxified)
    b: unpadded indices (jaxified)
    """
    return jax.lax.fori_loop(0, a.shape[0], lambda i, x: x.at[i].set(b[x[i]]), a)

def pad_L(L, nside, indices):
    """
    Pads a graph laplacian with rows and columns of zeros corresponding to masked pixels.
    --------
    L: sparse BCOO array
    nside: nside of the maps
    indices: unmasked indices
    --------
    returns BCOO array of shape (12*(nside**2),12*(nside**2))
    """
    data = L.data
    row = L.indices[:,0]
    col = L.indices[:,1]
    new_row = index_mapping_fun(row, indices)
    new_col = index_mapping_fun(col, indices)
    pad_L = jaxsparse.BCOO((L.data, jnp.column_stack((new_row, new_col))), shape=(12*(nside**2),12*(nside**2)))
    return pad_L

def block_padded_L(L, nside, nside_sup, indices):
    """
    converts the original graph laplacian into a block diagonal laplacian in the following format
    
           |L_0  0   0   0...   0 |
           | 0  L_1  0   0...   0 |
           | 0   0  L_2  0...   0 |
        L= | 0                  0 |   where N is the number of patches(superpixels.)
           | 0   .  ...         0 |
           | 0   .              0 |
           | 0   0   0   0...  L_N|
           
    and L_n is the padded graph laplacian for patch n. 
    --------------------------------------------------
    
    nside: nside of the full map
    nside_sup: nside of the superpixel map, in the above expression N=12*nside_sup**2
    indices: unmasked indices
    --------------------------------------------------
    returns: block diagonal L as in above, L for the superpixels    
    """
    num_patches = 12*(nside_sup**2)
    block_size = 12*(nside**2)//num_patches
    padded_L = pad_L(L, nside=nside, indices=indices)
    block_diag_L = block_diag(padded_L, block_size)
    return block_diag_L

def block_padded_L_v2(L, nside, nside_sup, indices):
    """
    converts the original graph laplacian into a block diagonal laplacian in the following format
    
           |L_0  0   0   0...   0 |
           | 0  L_1  0   0...   0 |
           | 0   0  L_2  0...   0 |
        L= | 0                  0 |   where N is the number of patches(superpixels.)
           | 0   .  ...         0 |
           | 0   .              0 |
           | 0   0   0   0...  L_N|
           
    and L_n is the padded graph laplacian for patch n. 
    --------------------------------------------------
    L: the original graph laplacian in jax BCOO format
    nside: nside of the full map
    nside_sup: nside of the superpixel map, in the above expression N=12*nside_sup**2
    indices: indices of full map where masked indices are set to -1
    --------------------------------------------------
    returns: block diagonal L as in above, L for the superpixels    
    """
    def block_diag(L, block_size):
        """
        Turns M into block diagonal format where each block is of shape (block_size, block_size)
        by putting zeros in places that do not belong to a block
        ----------
        M: jax BCOO array
        block_size: size of blocks
        """
        data = L.data
        idx = L.indices
        row = idx[:,0]
        col = idx[:,1]
        block_num = row//block_size
        blocks_delim_low = block_num*block_size 
        blocks_delim_high = block_num*block_size + block_size
        data = (blocks_delim_low<=(col))*((col)<blocks_delim_high)*data
        new_L = jaxsparse.BCOO((data, idx), shape=L.shape)
        return new_L

    def index_mapping_fun(a, b):
        """
        a: COO indices (jaxified)
        b: unpadded indices (jaxified)
        """
        return jax.lax.fori_loop(0, a.shape[0], lambda i, x: x.at[i].set(b[x[i]]), a)

    def pad_L(L, nside, indices):
        """
        Pads a graph laplacian with rows and columns of zeros corresponding to masked pixels.
        --------
        L: sparse BCOO array
        nside: nside of the maps
        indices: unmasked indices
        --------
        returns BCOO array of shape (12*(nside**2),12*(nside**2))
        """
        data = L.data
        row = L.indices[:,0]
        col = L.indices[:,1]
        new_row = index_mapping_fun(row, indices)
        new_col = index_mapping_fun(col, indices)
        pad_L = jaxsparse.BCOO((L.data, jnp.column_stack((new_row, new_col))), shape=(12*(nside**2),12*(nside**2)))
        return pad_L    
    
    num_patches = 12*(nside_sup**2)
    block_size = 12*(nside**2)//num_patches
    padded_L = pad_L(L, nside=nside, indices=indices)
    block_diag_L = block_diag(padded_L, block_size)
    return block_diag_L

def dot(x, y):
    return jnp.dot(x, y)
sp_dot = jaxsparse.sparsify(dot) #sparsified matmul, can even be done for tensordot/einsum
v_sp_dot = jax.vmap(sp_dot, in_axes=(None, 0))

@functools.partial(jax.jit, static_argnums=0)
def init_array(K, x):
    """
    Array initializer for chebyshev transformations.
    ----------
    :param K: order of the chebyshev polynomials.
    :param x: input array
    ----------
    returns: array of shape (N, K, M, F) where xout[:,0,...] = x
    """
    M, Fin = x.shape[-2], x.shape[-1]
    xout = jnp.empty((K, M, Fin))
    xout = xout.at[0].set(x)
    return xout #shape (K, M, Fin) where xout[0] = input.
        
v_init_array = jax.vmap(init_array, (None, 0), 0)

@functools.partial(jax.jit, static_argnums=0)
def transformer_init_array(K, x):
    """
    Array initializer for chebyshev transformations.
    ----------
    :param K: order of the chebyshev polynomials.
    :param x: input array
    ----------
    returns: array of shape (N, K, M, F) where xout[:,0,...] = x
    """    
    xout = jnp.empty_like(x)
    xout = jnp.expand_dims(xout, axis=1)
    xout = jnp.repeat(xout, K, axis=1, total_repeat_length=K)
    xout = xout.at[:,0,...].set(x)
    return xout

@jax.jit
def chebyshev_transform(K, L, x):
    """
    Recursively calculates [T_0(L)@x, T_1(L)@x, ..., T_{K-1}(L)@x]
    ---------
    :param K: int>=1, number of chebyshev polynomials T_k(L) to use. 
    :param L: the graph laplacian, jax.sparse BCOO of shape (M, M)
    :param x: input maps of shape (K, M, Fin) where x[0] = input
    ---------
    Returns [T_0(L)@x, T_1(L)@x, ..., T_K(L)@x], shape (K, M, Fin)
    """

    def K1branch(x):
        """
        The case when K=1
        """
        return x

    def Kbranch(x):
        """
        K>1 branch. 
        """
        def recursion(carry, _):
            """
            Chebyshev polynomials recursion rule. "_" is just a dummy variable.
            """
            y0, y1, = carry #set y0 = x[k-2], y1 = x[k-1] and use x[k] = 2*L@x[k-1] - x[k-2]
            y2 = 2*sp_dot(L,y1) - y0
            return (y1, y2), y2
        x = x.at[1].set(sp_dot(L,x[0])) #add L@input at K=1
        (_, _), xrest = jax.lax.scan(recursion, (x[0], x[1]), x[2:]) #calculate the rest
        x = x.at[2:].set(xrest)
        return x
            
    x = jax.lax.cond(K>1, lambda x: Kbranch(x), lambda x: K1branch(x), x) #(K, M, Fin)
    return x

v_chebyshev_transform = jax.vmap(chebyshev_transform, (None, None, 0), 0)
@jax.jit
def einsum(w, x):
    return jnp.einsum('ijk,jmi->mk', w, x)

v_einsum = jax.vmap(einsum, (None, 0), 0)

########----------------------------------abstract graph layers-------------------------------##########

class GraphChebyshevConv(nn.Module):
    """
    Abstract graph convolutional layer using the Chebyshev approximation.
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices
            and F are the number of input features.
    ------------------
    :param L: the graph laplacian in jax BCOO format
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param Fout: int >= 1, number of output features. output shape will be (B, M, Fout)
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output array of shape (N, M, Fout)
    """
    L: Array
    K: int
    Fout: int
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32        
    
    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel', self.kernel_init,
                            (inputs.shape[-1], self.K, self.Fout), #K x Fin x Fout
                            self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        #get input shape
        M, Fin = inputs.shape[-2], inputs.shape[-1]
        
        single_input = False
        if inputs.ndim == 2: #input (M, Fin)
            single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
            
        x = v_init_array(self.K, inputs) #shape (N, K, M, Fin)
        x = v_chebyshev_transform(self.K, self.L, x) #shape (N, K, M, Fin)
        x = v_einsum(kernel, x)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, 
                                  (self.Fout,),
                                  self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))
            
        if single_input:
            x = jnp.squeeze(x, axis=0)
        
        return x
    
class GraphDepthwiseChebyshevConv(nn.Module):
    """
    Abstract graph depthwise convolutional layer using the Chebyshev approximation.
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices
            and F are the number of input features.
    ------------------
    :param L: the graph laplacian in jax BCOO format
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param depth_multiplier: int >= 1, number of output features per input feature.
                             output shape will be (B, M, F*depth_multiplier)
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output array of shape (N, M, Fout)
    """
    L: Array
    K: int
    depth_multiplier: int
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32        
    
    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        kernel = self.param('kernel', self.kernel_init,
                            (inputs.shape[-1], self.depth_multiplier, 
                             self.K), #Fin x depth_mul x K
                            self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        #get input shape
        M, Fin = inputs.shape[-2], inputs.shape[-1]
        
        single_input = False
        if inputs.ndim == 2: #input (M, Fin)
            single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        x = v_init_array(self.K, inputs) #shape (N, K, M, Fin)
        x = v_chebyshev_transform(self.K, self.L, x) #shape (N, K, M, Fin)
        x = jnp.transpose(x, axes=(0,3,1,2)) #shape (N, Fin, K, M)
        x = jax.lax.dot_general(kernel, x, (((2,), (2,)), ((0), (1)))) # Fin x depth_mul x N x M
        x = jnp.reshape(x, (Fin*self.depth_multiplier, -1, M)) #Fin*depth_mul x N x M
        x = jnp.transpose(x, axes=(1,2,0)) #N x M x Fin*depth_mul
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, 
                                  (Fin*self.depth_multiplier,),
                                  self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))
        
        if single_input:
            x = jnp.squeeze(x, axis=0)
        
        return x
    
class GraphSeparableChebyshevConv(nn.Module):
    """
    Abstract graph separable convolutional layer using the Chebyshev approximation.
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices
            and F are the number of input features.
    ------------------
    :param L: the graph laplacian in jax BCOO format
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param Fout: number of output features
    :param depth_multiplier: int >= 1, number of output features per input feature for the depthwise step.
                             output shape of the depthwise step will be (B, M, F*depth_multiplier) 
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param pointwise_kernel_init: pointwise kernel initializer, defaults to 'he_normal'
    :param depthwise_kernel_init: depthwise kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output array of shape (N, M, Fout)
    """
    L: Array
    K: int
    Fout: int
    depth_multiplier: int = 1
    use_bias: bool = False
    pointwise_kernel_init: Callable = nn.initializers.he_normal()
    depthwise_kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32        
    
    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        d_kernel = self.param('depthwise_kernel', self.pointwise_kernel_init,
                              (inputs.shape[-1], self.depth_multiplier, 
                              self.K), #shape (Fin, depth_mul, K)
                              self.param_dtype)
        d_kernel = jnp.asarray(d_kernel, self.dtype)
        
        p_kernel = self.param('pointwise_kernel', self.depthwise_kernel_init,
                              (self.Fout, inputs.shape[-1]*self.depth_multiplier), #shape (Fout, Fin*depth_mul)
                              self.param_dtype)
        p_kernel = jnp.asarray(p_kernel, self.dtype)
        
        #get input shape
        M, Fin = inputs.shape[-2], inputs.shape[-1]
        
        single_input = False
        if inputs.ndim == 2: #input (M, Fin)
            single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        #depthwise step
        x = v_init_array(self.K, inputs) #shape (N, K, M, Fin)
        x = v_chebyshev_transform(self.K, self.L, x) #shape (N, K, M, Fin)
        x = jnp.transpose(x, axes=(0,3,1,2)) #shape (N, Fin, K, M)
        x = jax.lax.dot_general(d_kernel, x, (((2,), (2,)), ((0), (1)))) #shape (Fin, depth_mul, N, M)
        x = jnp.reshape(x, (Fin*self.depth_multiplier, -1, M)) #shape (Fin*depth_mul, N, M)
        #pointwise step
        x = jnp.transpose(x, axes=(1,0,2)) #shape (N, Fin*depth_mul, M)
        x = jnp.dot(p_kernel, x) #shape (Fout, N, M)
        x = jnp.transpose(x, axes=(1,2,0)) #shape (N, M, Fout)
        
        if self.use_bias:
            bias = self.param('bias', self.bias_init, 
                                      (self.Fout,),
                                      self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            x += jnp.reshape(bias, (1,) * (x.ndim - 1) + (-1,))
        
        if single_input:
            x = jnp.squeeze(x, axis=0)
        return x
    
    
###############-----v2--relies on regular conv after graph fft with chebyshev polynomials--------##############
class GraphChebyshevConv_v2(nn.Module):
    """
    Abstract graph convolutional layer using the Chebyshev approximation.
    Uses a regular convolution filter of kernel size (K, 4**p) after the initial chebyshev transformation.
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices
            and F are the number of input features.
    ------------------
    :param L: the graph laplacian in jax BCOO format
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param p: int >= 0, reduction factor. will reduce M by a factor of 4**p
              its behavior for general graphs are not guaranteed, this is mainly for
              healpix graphs.
    :param Fout: int >= 1, number of output features. output shape will be (B, M, Fout)
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output array of shape (N, M/(4**p), Fout)
    """
    L: Array
    K: int
    p: int 
    Fout: int
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32        
    
    
    def setup(self):
        self.filter = nn.Conv(features=self.Fout, kernel_size=(self.K, 4**self.p), strides=(1, 4**self.p),
                              padding='VALID', use_bias=self.use_bias, 
                              kernel_init=self.kernel_init, bias_init=self.bias_init,
                              dtype=self.dtype, param_dtype=self.param_dtype)    
    
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        #get input shape
        M, Fin = inputs.shape[-2], inputs.shape[-1]
        single_input = False
        if inputs.ndim == 2: #input (M, Fin)
            single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
            
        x = v_init_array(self.K, inputs) #shape (N, K, M, Fin)
        x = v_chebyshev_transform(self.K, self.L, x) #shape (N, K, M, Fin)
        
        x = self.filter(x) #shape ((N), 1, M, Fin), squeeze the K dim out
        x = jnp.squeeze(x, axis=-3)
        
        if single_input:
            x = jnp.squeeze(x, axis=0)
            
        return x
    
class GraphDepthwiseChebyshevConv_v2(nn.Module):
    """
    Abstract graph depthwise convolutional layer using the Chebyshev approximation. 
    Uses a regular convolution filter of kernel size (K, 4**p) after the initial chebyshev transformation.
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices
            and F are the number of input features.
    ------------------
    :param L: the graph laplacian in jax BCOO format
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param p: int >= 0, reduction factor. will reduce M by a factor of 4**p
              its behavior for general graphs are not guaranteed, this is mainly for
              healpix graphs.    
    :param depth_multiplier: int >= 1, number of output features per input feature.
                             output shape will be (B, M, F*depth_multiplier)
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output array of shape (N, M, Fout)
    """
    L: Array
    K: int
    p: int = 0
    depth_multiplier: int = 1
    use_bias: bool = False
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32        

    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        M, Fin = inputs.shape[-2], inputs.shape[-1]
        depth_conv = nn.Conv(features=self.depth_multiplier*Fin, 
                             kernel_size=(self.K, 4**self.p), 
                             strides=(1, 4**self.p),
                             padding='VALID', 
                             feature_group_count = Fin,
                             use_bias=self.use_bias, 
                             kernel_init=self.kernel_init, 
                             bias_init=self.bias_init,
                             dtype=self.dtype, 
                             param_dtype=self.param_dtype) 
        
        #check if inputs are batched
        single_input = False
        if inputs.ndim == 2: #input (M, Fin)
            single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
            
        x = v_init_array(self.K, inputs) #shape (N, K, M, Fin)
        x = v_chebyshev_transform(self.K, self.L, x) #shape (N, K, M, Fin)
            
        x = depth_conv(x)
        x = jnp.squeeze(x, axis=-3)
        
        if single_input:
            x = jnp.squeeze(x, axis=0)
            
        return x
    
class GraphSeparableChebyshevConv_v2(nn.Module):
    """
    Abstract graph separable convolutional layer using the Chebyshev approximation.
    Uses a regular convolution filter of kernel size (K, 4**p) after the initial chebyshev transformation.
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices
            and F are the number of input features.
    ------------------
    :param L: the graph laplacian in jax BCOO format
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param p: int >= 0, reduction factor. will reduce M by a factor of 4**p
              its behavior for general graphs are not guaranteed, this is mainly for
              healpix graphs.
    :param Fout: number of output features
    :param depth_multiplier: int >= 1, number of output features per input feature for the depthwise step.
                             output shape of the depthwise step will be (B, M, F*depth_multiplier) 
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param pointwise_kernel_init: pointwise kernel initializer, defaults to 'he_normal'
    :param depthwise_kernel_init: depthwise kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output array of shape (N, M, Fout)
    """
    L: Array
    K: int
    Fout: int
    p: int = 0
    depth_multiplier: int = 1
    use_bias: bool = False
    pointwise_kernel_init: Callable = nn.initializers.he_normal()
    depthwise_kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32        
    
    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)
        M, Fin = inputs.shape[-2], inputs.shape[-1]
        depth_conv = nn.Conv(features=self.depth_multiplier*Fin, 
                             kernel_size=(self.K, 4**self.p), 
                             strides=(1, 4**self.p),
                             padding='VALID', 
                             feature_group_count = Fin,
                             use_bias=False, 
                             kernel_init=self.depthwise_kernel_init, 
                             bias_init=self.bias_init,
                             dtype=self.dtype, 
                             param_dtype=self.param_dtype)
        
        point_conv = nn.Conv(features=self.Fout, 
                             kernel_size=(1, ), 
                             strides=(1, ),
                             padding='VALID', 
                             feature_group_count = 1,
                             use_bias=self.use_bias, 
                             kernel_init=self.pointwise_kernel_init, 
                             bias_init=self.bias_init,
                             dtype=self.dtype, 
                             param_dtype=self.param_dtype)
        
        #check if inputs are batched
        single_input = False
        if inputs.ndim == 2: #input (M, Fin)
            single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)
            
        x = v_init_array(self.K, inputs) #shape (N, K, M, Fin)
        x = v_chebyshev_transform(self.K, self.L, x) #shape (N, K, M, Fin)
            
        x = depth_conv(x)
        x = jnp.squeeze(x, axis=-3)
        x = point_conv(x)
        
        if single_input:
            x = jnp.squeeze(x, axis=0)
            
        return x


###########-----------------------------------Healpy Layers--------------------------------------###########
class HealpyPool(nn.Module):
    """
    Pooling layer for Healpy maps. Makes use of Healpy pixellization scheme and reduces the number of
    pixels by a factor of 4**p (a reduction in nside of a factor 2**p)
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    ---------------
    :param p: int>=1, reduction factor. 
    :param pool_type: either 'AVG' for average pooling or 'MAX' for max pooling
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.
    --------------
    returns: output of form {'nside': nside_in/2**p, 'indices': transformed_indices, 'maps': array of shape (N, M/4**p, F)}
             
    """
    p: int
    pool_type: str
    nside: Optional[int] = None
    indices: Optional[Array] = None
    
    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']

        outputs = {}
        nside_out = nside_in//(2**self.p)
        masked_indices, indices_out = _get_new_indices(indices_in, self.p, reduce=True)
        
        outputs['nside'] = nside_out
        outputs['indices'] = indices_out
        
        idx = _get_indices_idx(indices_in, masked_indices)
        
        x = jnp.take(x, idx, axis=-2) #input format (N)MF
        
        if self.pool_type == 'AVG':
            x = nn.avg_pool(x, (4**self.p, ), (4**self.p, ), padding='VALID')
        if self.pool_type == 'MAX':
            x = nn.max_pool(x, (4**self.p, ), (4**self.p, ), padding='VALID')
        
        outputs['maps'] = x
        
        return outputs
        
class HealpyPseudoConv(nn.Module): #seems to work. now we need to find a way construct a 
    """
    Convolutional Pooling layer for Healpy maps. Makes use of Healpy pixellization scheme and reduces the number of
    pixels by a factor of 4**p (a reduction in nside of a factor 2**p) by acting on the input maps with a 
    1D convolution of kernel size (4**p, )
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    ---------------
    :param p: int>=1, reduction factor.
    :param Fout: int>=1, number of output features
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param use_bias: whether to use bias or not
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel    
    --------------
    returns: output of form {'nside': nside_in/2**p, 'indices': transformed_indices, 'maps': array of shape (N, M/4**p, F)}
             
    """
    p: int                         
    Fout: int
    nside: Optional[int] = None
    indices: Optional[Array] = None
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32

    def setup(self):
        self.filter = nn.Conv(features=self.Fout, kernel_size=(4**self.p, ), strides=(4**self.p, ),
                              padding='VALID', use_bias=self.use_bias, 
                              kernel_init=self.kernel_init, bias_init=self.bias_init,
                              dtype=self.dtype, param_dtype=self.param_dtype)
    
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
        
        outputs = {}
        nside_out = nside_in//(2**self.p)
        masked_indices, indices_out = _get_new_indices(indices_in, self.p, reduce=True)
        outputs['nside'] = nside_out
        outputs['indices'] = indices_out
        
        idx = _get_indices_idx(indices_in, masked_indices)
        
        x = jnp.take(x, idx, axis=-2) #input format (N)MF
        x = self.filter(x)
        outputs['maps']=x
        
        return outputs
    
class HealpyPseudoConvTranspose(nn.Module):
    """
    Convolutional unpooling layer for Healpy maps. Makes use of Healpy pixellization scheme and increases the number of
    pixels by a factor of 4**p (a boost in nside of a factor 2**p) by acting on the input maps with a 
    1D transpose convolution of kernel size (4**p, )
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    ---------------
    :param p: int>=1, boost factor.
    :param Fout: int>=1, number of output features
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param use_bias: whether to use bias or not
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel    
    --------------
    returns: output of form {'nside': nside_in*2**p, 'indices': transformed_indices, 'maps': array of shape (N, M*4**p, F)}
             
    """    
    p: int
    Fout: int
    use_bias: bool
    nside: Optional[int] = None
    indices: Optional[Array] = None
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32         
    
    def setup(self):
        self.filter = nn.ConvTranspose(features=self.Fout, kernel_size=4**self.p, strides=4**self.p,
                                       padding='VALID', use_bias=self.use_bias, 
                                       kernel_init=self.kernel_init, bias_init=self.bias_init,
                                       dtype=self.dtype, param_dtype=self.param_dtype)
    
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
        
        outputs = {}
        nside_out = nside_in*(2**self.p)
        masked_indices, indices_out = _get_new_indices(indices_in, self.p, reduce=False)
        outputs['nside'] = nside_out
        outputs['indices'] = indices_out        
   
        x = self.filter(x)
        outputs['maps'] = x
        
        return outputs
    
##########---------------------------GraphConv Wrappers----------------------------------###########
class HealpyChebyshevConv(nn.Module): #works.
    """
    Healpy wrapper layer for the abstract graph chebyshev convolution layer.
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    --------------
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param Fout: int >= 1, number of output features. output shape will be (B, M, Fout)
    :param n_neighbors: int, number of neighbors for each pixel. used in calculating the 
                        graph laplacian. one of 8, 20, 40, 80
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output dict of form {'nside': nside_out, 'indices': indices_out, 'maps': (N, M, Fout)}  
    """
    K: int
    Fout: int
    n_neighbors: int
    use_bias: bool = True
    nside: Optional[int] = None
    indices: Optional[Array] = None
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
        
    @nn.nowrap
    def _get_layer(self, L):
        return GraphChebyshevConv(L=L, K=self.K, Fout=self.Fout, use_bias=self.use_bias,
                                  kernel_init=self.kernel_init, bias_init=self.bias_init,
                                  dtype=self.dtype, param_dtype=self.param_dtype) 
    
    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
   
        outputs = {}
        outputs['nside'] = nside_in
        outputs['indices'] = indices_in
        
        _L = get_L(nside=nside_in, indices=indices_in, n_neighbors=self.n_neighbors)
        
        graphconv = self._get_layer(L=_L)
        x = graphconv(x)
        
        outputs['maps'] = x
        
        return outputs

class HealpyDepthwiseChebyshevConv(nn.Module): #works.
    """
    Healpy wrapper layer for the abstract graph depthwise chebyshev convolution layer
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    --------------
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param depth_multiplier: int >= 1, number of output features per input feature. 
                             output shape will be (B, M, F*depth_multiplier)
    :param n_neighbors: int, number of neighbors for each pixel. used in calculating the 
                        graph laplacian.
    :param p: int>=0, reduction factor
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output dict of form {'nside': nside_in, 'indices': indices_in, 'maps': (N, M, F*depth_multiplier)}     
    """
    K: int
    depth_multiplier: int
    n_neighbors: int
    use_bias: bool = True
    nside: Optional[int] = None
    indices: Optional[Array] = None
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
        
    @nn.nowrap
    def _get_layer(self, L):
        return GraphDepthwiseChebyshevConv(L=L, K=self.K, depth_multiplier=self.depth_multiplier, 
                                           use_bias=self.use_bias, kernel_init=self.kernel_init, 
                                           bias_init=self.bias_init, dtype=self.dtype, 
                                           param_dtype=self.param_dtype)
     
    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            nside_in = self.nside
            indices_in = self.indices
            x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
            
        outputs = {}
        
        outputs['nside']=nside_in
        outputs['indices']=indices_in        
        
        _L = get_L(nside=nside_in, indices =indices_in, n_neighbors=self.n_neighbors)
        
        graphdepthconv = self._get_layer(L=_L)
        x = graphdepthconv(x)
        outputs['maps'] = x
        
        return outputs
    
class HealpySeparableChebyshevConv(nn.Module): #works.
    """
    Healpy wrapper layer for the abstract graph separable chebyshev convolution layer
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    --------------
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param Fout: int>=1, number of final output features. final output shape will be (B, M, Fout)
    :param depth_multiplier: int >= 1, number of output features per input feature in the depthwise step
                             intermediate output shape will be (B, M, F*depth_multiplier)
    :param n_neighbors: int, number of neighbors for each pixel. used in calculating the 
                        graph laplacian.
    :param p: int>=0, reduction factor
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param pointwise_kernel_init: kernel initializer, defaults to 'he_normal'
    :param depthwise_kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output dict of form {'nside': nside_in, 'indices': indices_in, 'maps': (N, M, F*depth_multiplier)}     
    """
    K: int
    Fout: int
    depth_multiplier: int = 1
    n_neighbors: int = 8
    use_bias: bool = True
    nside: Optional[int] = None
    indices: Optional[Array] = None
    pointwise_kernel_init: Callable = nn.initializers.he_normal()
    depthwise_kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
        
    @nn.nowrap
    def _get_layer(self, L):
        return GraphSeparableChebyshevConv(L=L, K=self.K, Fout=self.Fout,
                                           depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, 
                                           pointwise_kernel_init=self.pointwise_kernel_init, 
                                           depthwise_kernel_init=self.depthwise_kernel_init,
                                           bias_init=self.bias_init, dtype=self.dtype, 
                                           param_dtype=self.param_dtype)

    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
        
        outputs = {}
        
        outputs['nside'] = nside_in
        outputs['indices'] = indices_in         
        
        _L = get_L(nside=nside_in, indices=indices_in, n_neighbors=self.n_neighbors)
        
        graphseparableconv = self._get_layer(L=_L)
        x = graphseparableconv(x)
        outputs['maps'] = x
        
        return outputs
        
###############------------------------------wrappers for conv_v2----------------------------------###############

class HealpyChebyshevConv_v2(nn.Module): #works.
    """
    Healpy wrapper layer for the abstract graph chebyshev convolution layer using convolutions directly
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    --------------
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param Fout: int >= 1, number of output features. output shape will be (B, M, Fout)
    :param n_neighbors: int, number of neighbors for each pixel. used in calculating the 
                        graph laplacian.
    :param p: int>=0, reduction factor
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output dict of form {'nside': nside_out, 'indices': indices_out, 'maps': (N, M', Fout)}
             where M' = M/4**p, nside_out = nside_in/2**p and indices_out are valid indices after a reduction
             in nside by a factor of p
    """
    K: int
    Fout: int
    n_neighbors: int
    p: int = 0
    use_bias: bool = True
    nside: Optional[int] = None
    indices: Optional[Array] = None
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    
    @nn.nowrap
    def _get_layer(self, L):
        return GraphChebyshevConv_v2(L=L, K=self.K, p=self.p, Fout=self.Fout, use_bias=self.use_bias,
                                     kernel_init=self.kernel_init, bias_init=self.bias_init,
                                     dtype=self.dtype, param_dtype=self.param_dtype) 
    
    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
            
        outputs = {}
    
        nside_out = nside_in//(2**self.p)
        masked_indices, indices_out = _get_new_indices(indices_in, self.p, reduce=True)
        outputs['nside'] = nside_out
        outputs['indices'] = indices_out
        
        _L = get_L(nside=nside_in, indices=masked_indices, n_neighbors=self.n_neighbors)
        graphconv = self._get_layer(L=_L)
       
        idx = _get_indices_idx(indices_in, masked_indices)
        
        x = jnp.take(x, idx, axis=-2)
        x = graphconv(x)
        outputs['maps'] = x
        
        return outputs

class HealpyDepthwiseChebyshevConv_v2(nn.Module): #works.
    """
    Healpy wrapper layer for the abstract graph depthwise chebyshev convolution layer
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    --------------
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param depth_multiplier: int >= 1, number of output features per input feature. 
                             output shape will be (B, M, F*depth_multiplier)
    :param n_neighbors: int, number of neighbors for each pixel. used in calculating the 
                        graph laplacian.
    :param p: int>=0, reduction factor
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output dict of form {'nside': nside_out, 'indices': indices_out, 'maps': (N, M', Fout)}
             where Fout = Fin*depth_multiplier, M' = M/4**p, nside_out = nside_in/2**p 
             and indices_out are valid indices after a reduction
             in nside by a factor of 2**p    
    """
    K: int
    depth_multiplier: int
    n_neighbors: int
    p : int = 0
    use_bias: bool = True
    nside: Optional[int] = None
    indices: Optional[Array] = None
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
        
    @nn.nowrap
    def _get_layer(self, L):
        return GraphDepthwiseChebyshevConv_v2(L=L, K=self.K, p=self.p,
                                              depth_multiplier=self.depth_multiplier, 
                                              use_bias=self.use_bias, kernel_init=self.kernel_init, 
                                              bias_init=self.bias_init, dtype=self.dtype, 
                                              param_dtype=self.param_dtype)
     
    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            nside_in = self.nside
            indices_in = self.indices
            x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
            
        outputs = {}
        
        nside_out = nside_in//(2**self.p)
        reduced_indices, indices_out = _get_new_indices(indices_in, self.p, reduce=True)        
        
        outputs['nside']=nside_out
        outputs['indices']=indices_out        
        
        _L = get_L(nside=nside_in, indices=reduced_indices, n_neighbors=self.n_neighbors)
        graphdepthconv = self._get_layer(L=_L)
        
        idx = _get_indices_idx(indices_in, masked_indices)
        
        x = jnp.take(x, idx, axis=-2)
        outputs['maps'] = x
        
        return outputs
    
class HealpySeparableChebyshevConv_v2(nn.Module): #works.
    """
    Healpy wrapper layer for the abstract graph separable chebyshev convolution layer
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified.
    --------------
    :param K: int >= 1, number of chebyshev polynomials to use, K=K will use T_0, T_1, ..., T_{K-1}
    :param Fout: int>=1, number of final output features. final output shape will be (B, M, Fout)
    :param depth_multiplier: int >= 1, number of output features per input feature in the depthwise step
                             intermediate output shape will be (B, M, F*depth_multiplier)
    :param n_neighbors: int, number of neighbors for each pixel. used in calculating the 
                        graph laplacian.
    :param p: int>=0, reduction factor
    :param use_bias: bool, whether to use a bias vector or not, defaults to 'False'
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above.    
    :param pointwise_kernel_init: kernel initializer, defaults to 'he_normal'
    :param depthwise_kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    ------------------
    returns: output dict of form {'nside': nside_out, 'indices': indices_out, 'maps': (N, M', Fout)}
             where M' = M/4**p, nside_out = nside_in/2**p 
             and indices_out are valid indices after a reduction
             in nside by a factor of 2**p      
    """
    K: int
    Fout: int
    depth_multiplier: int = 1
    n_neighbors: int = 8
    p: int = 0
    use_bias: bool = True
    nside: Optional[int] = None
    indices: Optional[Array] = None
    pointwise_kernel_init: Callable = nn.initializers.he_normal()
    depthwise_kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
        
    @nn.nowrap
    def _get_layer(self, L):
        return GraphSeparableChebyshevConv_v2(L=L, K=self.K, 
                                              Fout=self.Fout,
                                              p=self.p,
                                              depth_multiplier=self.depth_multiplier, use_bias=self.use_bias, 
                                              pointwise_kernel_init=self.pointwise_kernel_init, 
                                              depthwise_kernel_init=self.depthwise_kernel_init,
                                              bias_init=self.bias_init, dtype=self.dtype, 
                                              param_dtype=self.param_dtype)

    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps']
        
        outputs = {}
        
        nside_out = nside_in//(2**self.p)
        reduced_indices, indices_out = _get_new_indices(indices_in, self.p, reduce=True)        
        
        outputs['nside']=nside_out
        outputs['indices']=indices_out        
        
        _L = get_L(nside=nside_in, indices=reduced_indices, n_neighbors=self.n_neighbors)
        graphseparableconv = self._get_layer(L=_L)
        
        idx = _get_indices_idx(indices_in, masked_indices)
        
        x = jnp.take(x, idx, axis=-2)
        outputs['maps'] = x
        
        return outputs    
    
################-------------------------------------------------------------------------------################

##########------------------------------------Transformer ideas--------------------------------------###########
####----------Abstract Tranformer-----------####
class GraphChebyshevTransformer(nn.Module): 
    """
    Abstract Graph transformer layer. Will break up the inputs super graphs each representing a patch of 
    the original graph, chebyshev transform each patch individually using a padded 
    (to take care of masked vertices), block diagonal graph laplacian. afterwards, it will
    act with a masked convolution of kernel_size = (K_patch, M//N_sup) = strides.
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices and F
            are the number of features.
    --------
    :param N_sup: number of vertices of the super graph, M%N_sup == 0
    :param L_p: graph laplacian for the patches, of block diagonal, padded form
    :param K_p: number of chebyshev polynomials to use for the "in patch" transformation
    :param Fout: number of output features
    :param indices: an array of vertex indices of shape (M, ) where masked vertices are set to -1
    :param use_bias: whether to use bias or not
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    --------
    outputs: array of shape (N, N_sup, Fout).
    """
    
    N_sup: int
    L_p: Array
    K_p: int
    Fout: int
    indices: Array
    use_bias: bool = True
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    einsum: bool = False
        
    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        #the wrapper will pass over padded maps to this layer

        inputs = jnp.asarray(inputs, self.dtype)  #input shape (B, M, F)
        M, F = inputs.shape[-2], inputs.shape[-1] #get input shape
        base = -jnp.ones(M)
        base = base.at[self.indices].set(self.indices)
        #calculate the bool mask.
        mask = jnp.where(base >= 0, 1, 0)
        #let's also figure out the percentage of unmasked pixels in each patch
        mask2 = jnp.reshape(mask, (self.N_sup, -1))
        occupancy = jnp.mean(mask2, axis=1)
        
        #now we can build the masked conv layer:
        maskedconv = MaskedConv(features=self.Fout, kernel_size=(self.K_p, M//self.N_sup), 
                                strides=(self.K_p, M//self.N_sup), padding='VALID', 
                                mask=mask, use_bias=self.use_bias, 
                                kernel_init=self.kernel_init, bias_init=self.bias_init,
                                dtype=self.dtype, param_dtype=self.param_dtype, einsum = self.einsum)
        
                                
        x = v_init_array(self.K_p, inputs) #shape (B, K_p, M, F)
        #chebyshev transform the patches
        x = v_chebyshev_transform(self.K_p, self.L_p, x) #shape (B, K_p, M, F)
        
        x = maskedconv(x)  #(B, N_sup, Fout) #do the conv
        print(x.shape)     
        
        return x

#####------------Healpy Wrapper-----------#####
class HealpyGraphTransformer(nn.Module):
    """
    Healpy wrapper for the abstract Graph transformer layer. Will break up the input maps into patches 
    of super pixels, chebyshev transform each patch individually using a padded 
    (to take care of masked vertices), block diagonal graph laplacian.
    will then chebyshev transform along the num_patches dim using a full graph laplacian. afterwards, it will
    act with a convolution of kernel (K_patch, K_super, 4**p).
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified. for the transformer, indices_in have to be padded, i.e indices[masked_pixels] = -1.
    --------
    :param N_s: nside of the super pixel map
    :param K_p: number of chebyshev polynomials to use for the "in patch" transformation
    :param n_neighbors_p: n_neighbors for the in-patch graph laplacian
    :param Fout: number of output features
    :param use_bias: whether to use bias or not
    :param threshold: optional float between 0 and 1. if provided, will calculate a boolean mask for the 
                      superpixels. a superpixel will be masked if it has less than 100*threshold percent
                      of its sub pixels masked. useful for attention weight masking.
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above. 
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    --------
    returns: output dict of form {'nside': nside_out, 'indices': indices_out, 'maps': (N, M', Fout)}
             where M' = M/4**p, nside_out = nside_in/2**p and indices_out are valid indices after a reduction
             in nside by a factor of p
    """
    nside_sup: int
    K_p: int
    Fout: int
    n_neighbors_p: int
    use_bias: bool = True
    threshold: Optional[float] = 0
    nside: Optional[int] = None
    indices: Optional[Array] = None
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    einsum: bool = False
        
    @nn.nowrap
    def _get_layer(self, indices_in, L_p):
        return GraphChebyshevTransformer(N_sup=12*(self.nside_sup**2), L_p=L_p,
                                         K_p=self.K_p, Fout=self.Fout, 
                                         indices=indices_in, 
                                         use_bias=self.use_bias, 
                                         kernel_init=self.kernel_init,
                                         bias_init=self.bias_init, dtype=self.dtype, 
                                         param_dtype=self.param_dtype,
                                         einsum=self.einsum)
    
    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps'] #padded indices are not jittable
                                                                                         #indices are unpadded
        nside_out = self.nside_sup
        indices_out = np.arange(12*(nside_out**2))
        
        N_sup = 12*self.nside_sup**2 #number of patches


        #prepare padded input to the graph layer:
        graph_in = jnp.empty((x.shape[0], 12*nside_in**2, x.shape[2]))
        #scatter the input maps onto the empty array
        graph_in = graph_in.at[...,indices_in,:].set(x)        
        #get the graph laplacians
        base_L = get_L(nside=nside_in, indices=indices_in, n_neighbors=self.n_neighbors_p)
        L_p = block_padded_L(L=base_L, nside=nside_in, nside_sup=self.nside_sup, indices=indices_in)
        
        #get the graph layer:
        graphtransformer = self._get_layer(indices_in=indices_in, L_p=L_p)
        #act with the transformer:
        graph_out = graphtransformer(graph_in)
        
        #construct the output dict:
        #this is only there if we want to act with other graph convolutions
        outputs = {}
        outputs['nside'] = nside_out
        outputs['indices'] = indices_out
        outputs['maps'] = graph_out
        
        if self.threshold: #do not jit this part or change it to device arrays
                           #since the shape is data dependent
            assert 0<= self.threshold <=1, f'superpix_mask_threshold must be between 0 and 1'
            bool_mask = np.zeros(12*nside_in**2)
            bool_mask[indices_in] = 1
            bool_mask = np.reshape(bool_mask, (N_sup, -1)) #bool mask per patch
            occupancy = np.mean(bool_mask, axis=1)
            bool_mask_super = np.where(occupancy >= self.threshold, 1, 0) 
            outputs['mask'] = bool_mask_super
            outputs['indices'] = outputs['indices'][bool_mask_super > 0]
            outputs['maps'] = outputs['maps'][:, bool_mask_super > 0, :]
        
        return outputs


    
class MaskedConv(nn.Module):
    """
    Masked Convolution. Currently implemented for the very limited case 
    where inputs have 2 spatial dims, kernel_size = strides and input_spatial_dim % kernel_size = 0
    moreover, input_spatial_dim[0] = kernel_size[0] although that can be relaxed fairly easily.
    input_spatial_dims correspond to input.shape[1], input_shape[2] assuming input of shape (B, H, W, C)
    This implementation is mainly for use in a graph convolutional mlp-mixer like scenario where 
    this module will be used to create embeddings as in the original mlp-mixer paper.
    Because of the targeted implementation, this is actually designed to act on the output 
    of the chebyshev transform of a 1 dimensional 'image' representing the graph vertices.
    -----------
    :param features: number of output features.
    :param kernel_size: size of the convolution kernel, currently limited to the case described above.
    :param strides: strides of the kernel, currently limited to kernel_size=strides.
    :param padding: either 'VALID' or 'SAME'
    :param mask: boolean mask with the same shape as input_spatial_dims[-1], the mask will be repeat along 
                 input_spatial_dims[-2] and broadcasted to the shape of the convolution kernel
    :param input_dilation: dilation to apply to inputs. currently limited to 1
    :param kernel_dilation: dilation to apply to the kernel. currently limited to 1
    :param feature_group_count: number of groups to divide the input features into. 
                                inputs[-1] % feature_group_count == 0.
    :param use_bias: whether to apply bias to the output or not.
    :param kernel_init: initializer for the convolution kernel
    :param bias_init: initializer for the bias
    :param dtype: dtype of the input data
    :param param_dtype: dtype of the kernels
    :param einsum: whether to do a scanned conv or use einsum to multiply the input with the kernel.
                   mainly for testing. einsum seems to run OOM
    :param precision: precision of the output
    """
    features: int
    kernel_size: Sequence[int]
    strides: Union[None, int, Sequence[int]] = 1
    padding: PaddingLike = 'SAME'
    mask: Array = None
    input_dilation: Union[None, int, Sequence[int]] = 1
    kernel_dilation: Union[None, int, Sequence[int]] = 1
    feature_group_count: int = 1
    use_bias: bool = True
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.he_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    einsum: bool = False
    precision: PrecisionLike = None

        
    @nn.compact
    def __call__(self, inputs: Array) -> Array:
        """
        Applies a shared, masked convolution to the inputs.
        Masking is achieved by multiplying the kernel with a boolean mask before the convolution step
        and the convolution is scanned over patches of the inputs where patches correspond to
        valid positions of the convolution kernel in the input space.
        -----------
        
        inputs: input data with dimensions (batch, spatial_dims..., features).
        This is the channels-last convention, i.e. NHWC for a 2d convolution
        and NDHWC for a 3D convolution. Note: this is different from the input
        convention used by `lax.conv_general_dilated`, which puts the spatial
        dimensions last.
        -----------
        Returns:
        The convolved data.
        """    
        if isinstance(self.kernel_size, int):
            raise TypeError('Expected Conv kernel_size to be a'
                            ' tuple/list of integers (eg.: [3, 3]) but got'
                            f' {self.kernel_size}.')
        else:
            kernel_size = tuple(self.kernel_size)

        def maybe_broadcast(x: Optional[Union[int, Sequence[int]]]) -> (Tuple[int, ...]):
            if x is None:
            # backward compatibility with using None as sentinel for
            # broadcast 1
                x = 1
            if isinstance(x, int):
                return (x,) * len(kernel_size)
            return tuple(x)

        is_single_input = False
        if inputs.ndim == len(kernel_size) + 1:
            is_single_input = True
            inputs = jnp.expand_dims(inputs, axis=0)

        # self.strides or (1,) * (inputs.ndim - 2)
        strides = maybe_broadcast(self.strides)
        input_dilation = maybe_broadcast(self.input_dilation)
        kernel_dilation = maybe_broadcast(self.kernel_dilation)
        
        padding_lax = canonicalize_padding(self.padding, len(kernel_size))
        
        #get input shape:
        K, M, Fin = inputs.shape[-3], inputs.shape[-2], inputs.shape[-1]
        
        #input maps are of shape (B, K, M, F)
        #kernel size is (K, M/N) and M%N == 0
        
        
        assert M % kernel_size[-1] == 0 #the kernel size needs to be so that
        assert K % kernel_size[-2] == 0 #we can create N patches of shape (K, M/N)
        assert K // kernel_size[-2] == 1 #without any leftovers
        # One shared convolutional kernel for all pixels in the output.
        # kernel size is a tuple.
        assert Fin % self.feature_group_count == 0
        kernel_shape = kernel_size + (Fin // self.feature_group_count, self.features)
        kernel = self.param('kernel', self.kernel_init, kernel_shape, self.param_dtype)
        kernel = jnp.asarray(kernel, self.dtype)
        
        #N are the number of patches of the input, each patch corresponding to a valid
        #position of the convolution kernel with respect to the input space
        x = jnp.reshape(inputs, (-1, K, M//kernel_size[-1], kernel_size[-1], Fin)) #reshape the input to (B, K, N, M/N, F)
        x = jnp.transpose(x, axes=(0, 2, 1, 3, 4)) #   (B, N, K, M/N, F)
        mask = jnp.reshape(self.mask, (-1, kernel_size[-1])) #(N, M/N)
        mask = jnp.expand_dims(mask, axis=1) #(N, K, M/N)
        mask = jnp.expand_dims(mask, axis=-1) #(N, K, M/N, fin/group*fout) #this is just to make the mask
        mask = jnp.expand_dims(mask, axis=-1).astype(np.float32) #(N, K, M/N, fin/group, fout) #broadcastable to kernel
                           
        if self.einsum == True: #doesn't fit in memory but scan seems to be slow.
            masked_kernel = kernel*mask
            path = jnp.einsum_path('...kmc,...kmcf->...f',x, masked_kernel,optimize='optimal')
            y = jnp.einsum('...kmc,...kmcf->...f',x, masked_kernel,optimize=path[0])
        
        elif self.einsum == False:
            def _conv_dimension_numbers(input_shape):
                """Computes the dimension numbers based on the input shape."""
                ndim = len(input_shape)
                lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
                rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
                out_spec = lhs_spec
                return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

            #when we are calculating dimension numbers, we need to calculate them ignoring the 
            #added patch number dimension.
            act_input_shape = (x.shape[0], x.shape[2], x.shape[3], x.shape[4]) #(B, K, M/N, fin)
            dimension_numbers = _conv_dimension_numbers(act_input_shape)

            #now we have the proper dimension numbers, we can actually do the convolution.

            #maybe einsum would be faster
            #input shape: (B, N, K, M/N, fin)
            #mask: (N, 1, M/N, 1, 1)
            #kernel: (K, M/N, fin, fout)
            #without the mask, you would do jnp.einsum('bnkmc,kmcf->bnf',input,kernel)
            #if broadcastable, i could do kernel -> kernel*mask, shape (N, K, M/N, fin,fout)
                              #jnp.einsum('...kmc,...kmnf->...f', input, kernel)
            #path = jnp.einsum_path('...kmc,...kmnf->...f', input, kernel, optimize='optimal')[0]
            #out 

            def scanned_conv(x, kernel, mask):
                """
                scans over the inputs and the mask and multiplies the kernel with the mask before doing
                the convolution.
                """
                def body_func(carry, mask):
                    i, kernel = carry
                    kernel = kernel*mask
                    y = jax.lax.conv_general_dilated(lhs=x[:,i,...], #at each loop iteration, convolve mask*kernel
                                                     rhs=kernel,     #with the corresponding patch.
                                                     window_strides=strides, #strides actually don't matter,
                                                     padding=padding_lax, #kernel_size=patch_size anyway #padding doesn't matter 
                                                     lhs_dilation=input_dilation,
                                                     rhs_dilation=kernel_dilation,
                                                     dimension_numbers=dimension_numbers, #modified dimension numbers
                                                     feature_group_count=self.feature_group_count, #where we have ignored
                                                     precision=self.precision) #the added patch dimension.
                    return (i+1, kernel), y
                return jax.lax.scan(body_func, (0, kernel), mask)

            (_, _), y = scanned_conv(x=x, kernel=kernel, mask=mask) #output is (N, B, 1, 1, F)

            #print('out shape after scannedconv ', y.shape)
            y = jnp.squeeze(y, axis=(-3, -2)) #(N, B, F).
            y = jnp.transpose(y, axes=(1,0,2))  #(B, N, F)
        
        if self.use_bias:
            bias_shape = (self.features,)
            bias = self.param('bias', self.bias_init, bias_shape, self.param_dtype)
            bias = jnp.asarray(bias, self.dtype)
            bias = bias.reshape((1,) * (y.ndim - bias.ndim) + bias.shape)
            y += bias
        
        return y

########-----------------------Attention basics--------------------------#########

def scaled_dot_product_attention_adjacency(query: Array, 
                                           key: Array, 
                                           value: Array, 
                                           adjacency_matrix: Array,
                                           dropout_rng: Optional[PRNGKey] = None,
                                           dropout_rate: float = 0.,
                                           broadcast_dropout: bool = True,
                                           deterministic: bool = False,
                                           dtype: Dtype = jnp.float32,
                                           precision: PrecisionLike = None):
    """
    Calculate the attention weights.
    query, key, value must have matching leading dimensions.
    key, value must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask is given as a 1D array of bool corresponding to the masked positions
    ---------------
    :param query: query shape == (..., seq_len_q, num_heads, depth)
    :param key: key shape == (..., seq_len_k, num_heads, depth)
    :param value: value shape == (..., seq_len_v, num_heads, depth_v)
    :param adjacency_matrix: graph adjacency matrix in sparse BCOO format
    :param dropout_rng: prng key for dropout
    :param dropout_rate: dropout rate
    :param broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    :param dtype: dtype for calcualations
    :param precision: precision of the calcualtions
    ---------------
    Returns:
    output attention matrix
    Function taken from:
    https://www.tensorflow.org/text/tutorials/transformer
    https://flax.readthedocs.io/en/latest/_modules/flax/linen/attention.html
    """
    
    
    # shape for scale and matrix
    query_depth = query.shape[-1]
    seq = query.shape[-3]
    print(seq, 'num seq')
    key_depth = key.shape[-1]
    A = adjacency_matrix
    idx = A.indices #graph adjacency indices, assuming A is sparse
    
    query_part = jnp.take(query, idx[:,0], axis=1)/jnp.sqrt(key_depth).astype(dtype) #input is (batch, seqq', num_heads, feat),
                                                                                      #seqq' = idx[:,0].shape
    print(query_part.shape, 'shape after embedding lookup')
    key_part = jnp.take(key, idx[:,1], axis=1) #input is (batch, seqk', num_heads, feat)
    
    #replace with einsum to get (b, h, seq', seq') and then scale using sqrt(feat)
    #qk = jnp.sum(query_part*key_part, axis=-1, keepdims=True)/jnp.sqrt(key_depth).astype(dtype) #(batch, seq, head, 1)
    #print(query_part.shape, 'shape of qk')
    
    weights_part = jnp.einsum('...qhd,...khd->...hqk', query_part, key_part, precision=precision) #(b, h, seqq', seqk')
    print('shape of weights_part', weights_part.shape)
    weights_part = jnp.transpose(weights_part, axes=(2, 3, 0, 1)) #(seqq', seqk', b, h)
    print('shape of weights_part', weights_part.shape)
    weights_part = jax.ops.segment_sum(data=weights_part, segment_ids=idx[:,0], num_segments=seq) #(seqq, seqk', b, h)
    print('shape of weights_part', weights_part.shape)
    weights_part = jnp.transpose(weights_part, axes=(1, 0, 2, 3)) #(seqk', seqq, b, h)
    print('shape of weights_part', weights_part.shape)
    weights_part = jax.ops.segment_sum(data=weights_part, segment_ids=idx[:,0], num_segments=seq) #(seqk, seqq, b, h)
    print('shape of weights_part', weights_part.shape)
    weights = jnp.transpose(weights_part, axes=(2, 3, 1, 0)) #(b, h, seqq, seqk)
    print('shape of attn_weights', weights.shape)
    
    value_part = jnp.take(value, idx[:,1], axis=1) #sequence dim. (batch, seq', num_heads, feat)
    print('shape of value_part', value_part.shape)
    value_part = jnp.transpose(value_part, axes=(1, 0, 2, 3)) #(seq', batch, heads, feat)
    print('shape of value_part', value_part.shape)
    value_sum = jax.ops.segment_sum(data=value_part, segment_ids=idx[:,0], num_segments=seq) #(seq, batch, heads, feat)
    print('shape of value_sum', value_sum.shape)
    value_sum = jnp.transpose(value_sum, axes=(1, 0, 2, 3)) #(batch, seq, heads, feat)                         
    
    weights = jax.nn.softmax(weights, axis=-1).astype(dtype)
                            
    if not deterministic and dropout_rate > 0.:
        keep_prob = 1.0 - dropout_rate
        if broadcast_dropout:
            # dropout is broadcast across the batch + head dimensions
            dropout_shape = tuple([1] * (key.ndim - 2)) + weights.shape[-2:]
            keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
        else:
            keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
        multiplier = (keep.astype(output.dtype) / jnp.asarray(keep_prob, dtype=dtype))
        weights = weights * multiplier
    
    output = jnp.einsum('...hqk,...khd->...qhd', weights, value_sum, precision=precision)
    
    #instead of unscaled softmax, we can einsum -> scale -> segmentsum -> mask -> softmax to get the weights
    #decide on the scaling, scale before segment
    # get the unscaled softmax
    #unscaled_softmax = jnp.exp(qk)
    #weighted_values = value_part * unscaled_softmax

    # get the weights, #segment sum works on the 1st axis only.
    #unscaled_softmax = jnp.transpose(unscaled_softmax, axes=(1,0,2,3)) #(seq, batch, head, depth)
    
    #weighted_values = jnp.transpose(weighted_values, axes=(1,0,2,3)) #(seq, batch, head, depth)
    
    #softmax_sum = jax.ops.segment_sum(data=unscaled_softmax, segment_ids=idx[:,0], num_segments=seq)
    #print(unscaled_softmax.shape, 'shape of qk after segment sum')
    #value_sum = jax.ops.segment_sum(data=weighted_values, segment_ids=idx[:,0], num_segments=seq)
    
    #softmax_sum = jnp.transpose(softmax_sum, axes=(1,0,2,3)) #(batch, seq, head, depth)
    #value_sum = jnp.transpose(value_sum, axes=(1,0,2,3))
    
    #output = value_sum/softmax_sum #shape (batch, seq, head, depth)
    
    #if not deterministic and dropout_rate > 0.:
    #    keep_prob = 1.0 - dropout_rate
    #    if broadcast_dropout:
    #        # dropout is broadcast across the batch + head dimensions
    #        dropout_shape = tuple([1] * (output.ndim - 3)) + output.shape[-3:-2] + tuple([1] * (output.ndim - 2))
    #        keep = jax.random.bernoulli(dropout_rng, keep_prob, dropout_shape)
    #    else:
    #        keep = jax.random.bernoulli(dropout_rng, keep_prob, output.shape)
    #    multiplier = (keep.astype(output.dtype) / jnp.asarray(keep_prob, dtype=dtype))
    #    output = output * multiplier
    #print(output.shape, 'shape of attention_fn')
    
    return output

    
class MultiHeadAdjacenyAttention(nn.Module):
    """
    Multi-head dot-product attention with adjacency matrix for graphs
    ----------
    :param num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
                       should be divisible by the number of heads.
    :param A: graph adjacency matrix in sparse BCOO format
    :param mask: boolean mask of shape (inputs_q.shape[-2])
    :param dtype: the dtype of the computation (default: float32)
    :param param_dtype: the dtype passed to parameter initializers (default: float32).
    :param qkv_features: dimension of the key, query, and value.
    :param out_features: dimension of the last projection
    :param dropout_rate: dropout rate
    :param broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    :param deterministic: if false, the attention weight is masked randomly
                          using dropout, whereas if true, the attention weights
                          are deterministic.
    :param precision: numerical precision of the computation see `jax.lax.Precision`
                      for details.
    :param kernel_init: initializer for the kernel of the Dense layers.
    :param bias_init: initializer for the bias of the Dense layers.
    :param use_bias: bool: whether pointwise QKVO dense transforms use bias.
    ----------
    returns: output of shape (batch_sizes..., length, features)
    """
    num_heads: int
    A: Array
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    dropout_rate: float = 0.
    broadcast_dropout: bool = True
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.he_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_bias: bool = True
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
    
    @nn.compact
    def __call__(self,
                 inputs_q: Array,
                 inputs_kv: Array,
                 deterministic: Optional[bool] = None):
        """Applies multi-head dot product attention on the input data.

        Projects the inputs into multi-headed query, key, and value vectors,
        applies dot-product attention and project the results to an output vector.

        
        :param inputs_q: input queries of shape (batch_sizes..., length, features)
        :param inputs_kv: key/values of shape (batch_sizes..., length, features)
        :param deterministic: if false, the attention weight is masked randomly
                              using dropout, whereas if true, the attention weights
                              are deterministic.

        Returns: output of shape (batch_sizes..., length, features)
        """
        features = self.out_features or inputs_q.shape[-1]
        qkv_features = self.qkv_features or inputs_q.shape[-1]
        assert qkv_features % self.num_heads == 0, ('feature dimension must be divisible by number of heads.')
        head_dim = qkv_features // self.num_heads

        dense = functools.partial(nn.DenseGeneral, axis=-1,
                                                   dtype=self.dtype,
                                                   param_dtype=self.param_dtype,
                                                   features=(self.num_heads, head_dim),
                                                   kernel_init=self.kernel_init,
                                                   bias_init=self.bias_init,
                                                   use_bias=self.use_bias,
                                                   precision=self.precision)
        # project inputs_q to multi-headed q/k/v
        # dimensions are then [batch..., length, n_heads, n_features_per_head]
        query, key, value = (dense(name='query')(inputs_q),
                             dense(name='key')(inputs_kv),
                             dense(name='value')(inputs_kv))
    
    
        dropout_rng = None
        if self.dropout_rate > 0.:  # Require `deterministic` only if using dropout.
            m_deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
            if not m_deterministic:
                dropout_rng = self.make_rng('dropout')
        else:
            m_deterministic = True    
    
        # apply attention
        x = scaled_dot_product_attention_adjacency(query=query,
                                                   key=key,
                                                   value=value,
                                                   adjacency_matrix=self.A,
                                                   dropout_rng=dropout_rng,
                                                   dropout_rate=self.dropout_rate,
                                                   broadcast_dropout=self.broadcast_dropout,
                                                   deterministic=m_deterministic,
                                                   dtype=self.dtype,
                                                   precision=self.precision)
        print(x.shape)
        # back to the original inputs dimensions
        out = nn.DenseGeneral(features=features,
                              axis=(-2, -1),
                              kernel_init=self.kernel_init,
                              bias_init=self.bias_init,
                              use_bias=self.use_bias,
                              dtype=self.dtype,
                              param_dtype=self.param_dtype,
                              precision=self.precision,
                              name='out')(x)
        print(out.shape, 'shape of output mha module')
        return out    

class HealpyMultiHeadAttention(nn.Module):
    """
    Multi-head dot-product attention with adjacency matrix for graphs
    ----------
    :param num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
                      should be divisible by the number of heads.
    :param qkv_features: dimension of the key, query, and value.
    :param out_features: dimension of the last projection
    :param n_neighbors: n_neighbors for the adjacency matrix
    :param dropout_rate: dropout rate for attention weights
    :param broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    :param deterministic: if false, the attention weight is masked randomly
                          using dropout, whereas if true, the attention weights
                          are deterministic.
    :param kernel_init: initializer for the kernel of the Dense layers.
    :param bias_init: initializer for the bias of the Dense layers.
    :param use_bias: bool: whether pointwise QKVO dense transforms use bias.
    :param dtype: the dtype of the computation (default: float32)
    :param param_dtype: the dtype passed to parameter initializers (default: float32).
    :param precision: numerical precision of the computation see `jax.lax.Precision`
                      for details.
    ----------
    returns: output of shape (batch_sizes..., length, features)
    """    
    num_heads: int
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    n_neighbors: int = 8
    class_token: bool = False
    dropout_rate: float = 0.
    broadcast_dropout: bool = True
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.he_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_bias: bool = True    
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
        
    @nn.nowrap
    def _get_layer(self, A):
        return MultiHeadAdjacenyAttention(num_heads=self.num_heads, 
                                          A=A,
                                          qkv_features=self.qkv_features, 
                                          out_features=self.out_features,
                                          dropout_rate=self.dropout_rate,
                                          broadcast_dropout=self.broadcast_dropout,
                                          deterministic=self.deterministic,
                                          kernel_init=self.kernel_init,
                                          bias_init=self.bias_init, 
                                          use_bias=self.use_bias,
                                          dtype=self.dtype, 
                                          param_dtype=self.param_dtype,
                                          precision=self.precision)
    
    @nn.compact
    def __call__(self, inputs):
        nside_in, indices_in, maps = inputs['nside'], inputs['indices'], inputs['maps']
        A = get_A(nside=nside_in, indices=indices_in, n_neighbors=self.n_neighbors) #get the adjacency matrix
        if class_token == True:
            #add a row and a column of 1s corresponding to the class token
            A = A.todense()
            B = jnp.zeros((A.shape[0]+1, A.shape[1]+1), dtype=A.dtype)
            B.at[1,1].set(A)
            B.at[0,:].set(1)
            B.at[:,0].set(1)
            B.at[0,0].set(0)
            A = jaxsparse.BCOO.fromdense(B)
            
        mha = self._get_layer(A=A) #get the multi head adjacency attention module
        out = mha(maps, maps) #this is technically self attention
        #we don't really need to repack it into a dict since no other layer after attention will
        #use nside and indices info but we might want to keep it just in case.
        output = {}
        output['nside'] = nside_in
        output['indices'] = indices_in
        output['maps'] = out
            
        return output
    
class HealpyMultiHeadAttention_v2(nn.Module):
    """
    Multi-head dot-product attention with adjacency matrix for graphs
    ----------
    :param num_heads: number of attention heads. Features (i.e. inputs_q.shape[-1])
                      should be divisible by the number of heads.
    :param qkv_features: dimension of the key, query, and value.
    :param out_features: dimension of the last projection
    :param n_neighbors: n_neighbors for the adjacency matrix
    :param dropout_rate: dropout rate for attention weights
    :param broadcast_dropout: bool: use a broadcasted dropout along batch dims.
    :param deterministic: if false, the attention weight is masked randomly
                          using dropout, whereas if true, the attention weights
                          are deterministic.
    :param kernel_init: initializer for the kernel of the Dense layers.
    :param bias_init: initializer for the bias of the Dense layers.
    :param use_bias: bool: whether pointwise QKVO dense transforms use bias.
    :param dtype: the dtype of the computation (default: float32)
    :param param_dtype: the dtype passed to parameter initializers (default: float32).
    :param precision: numerical precision of the computation see `jax.lax.Precision`
                      for details.
    ----------
    returns: output of shape (batch_sizes..., length, features)
    """    
    num_heads: int
    qkv_features: Optional[int] = None
    out_features: Optional[int] = None
    n_neighbors: int = 8
    class_token: bool = False
    dropout_rate: float = 0.
    broadcast_dropout: bool = True
    deterministic: Optional[bool] = None
    kernel_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.he_normal()
    bias_init: Callable[[PRNGKey, Shape, Dtype], Array] = nn.initializers.zeros
    use_bias: bool = True    
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32
    precision: PrecisionLike = None
        
    @nn.compact
    def __call__(self, inputs):
        nside_in, indices_in, maps = inputs['nside'], inputs['indices'], inputs['maps']
        A = get_A(nside=nside_in, indices=indices_in, n_neighbors=self.n_neighbors) #get the adjacency matrix
        nse = A.nse
        if self.class_token == True:
            #add a row and a column of 1s corresponding to the class token
            A = A.todense()
            B = jnp.zeros((A.shape[0]+1, A.shape[1]+1), dtype=A.dtype)
            B.at[1:,1:].set(A)
            B.at[0,:].set(1)
            B.at[:,0].set(1)
            B.at[0,0].set(0)
            nse = nse + 2*(A.shape[0]+1)-1
            A = jaxsparse.BCOO.fromdense(B, nse=nse)
        
        #mha = functools.partial(nn.MultiHeadDotProductAttention, num_heads=self.num_heads,
        #                                                         dtype=self.dtype,
        #                                                         param_dtype=self.param_dtype,
        #                                                         qkv_features=self.qkv_features,
        #                                                         out_features=self.out_features,
        #                                                         broadcast_dropout=self.broadcast_dropout,
        #                                                         dropout_rate=self.dropout_rate,
        #                                                         deterministic=self.deterministic,
        #                                                         precision=self.precision,
        #                                                         kernel_init=self.kernel_init,
        #                                                         bias_init=self.bias_init,
        #                                                         use_bias=self.use_bias)
        
        #maps shape (batch, seq, feat)
        idx = A.indices
        
        q = jnp.take(maps, idx[:,0], axis=1)
        kv = jnp.take(maps, idx[:,1], axis=1)
        
        out = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                                                                 dtype=self.dtype,
                                                                 param_dtype=self.param_dtype,
                                                                 qkv_features=self.qkv_features,
                                                                 out_features=self.out_features,
                                                                 broadcast_dropout=self.broadcast_dropout,
                                                                 dropout_rate=self.dropout_rate,
                                                                 deterministic=self.deterministic,
                                                                 precision=self.precision,
                                                                 kernel_init=self.kernel_init,
                                                                 bias_init=self.bias_init,
                                                                 use_bias=self.use_bias)(q, kv)
        #out = mha(q, kv) #shape (batch, seq', out_feat), seq' = A.indices length
        out = jnp.transpose(out, axes=(1,0,2)) #shape (seq', batch, out_feat)
        out = jax.ops.segment_sum(data=out, segment_ids=idx[:,0], num_segments=maps.shape[1]) #shape (seq, batch, out_feat)
        out = jnp.transpose(out, axes=(1,0,2)) #shape (batch, seq, out_feat)
        
        output = {}
        output['nside'] = nside_in
        output['indices'] = indices_in
        output['maps'] = out
            
        return output
                
#########-----------------------masked_conv helper functions------------------------#########
def _conv_dimension_numbers(input_shape):
    """
    Computes the dimension numbers based on the input shape.
    """
    ndim = len(input_shape)
    lhs_spec = (0, ndim - 1) + tuple(range(1, ndim - 1))
    rhs_spec = (ndim - 1, ndim - 2) + tuple(range(0, ndim - 2))
    out_spec = lhs_spec
    return jax.lax.ConvDimensionNumbers(lhs_spec, rhs_spec, out_spec)

def canonicalize_padding(padding: PaddingLike, rank: int) -> LaxPadding:
    """"
    Canonicalizes conv padding to a jax.lax supported format.
    """
    if isinstance(padding, str):
        return padding
    if isinstance(padding, int):
        return [(padding, padding)] * rank
    if isinstance(padding, Sequence) and len(padding) == rank:
        new_pad = []
        for p in padding:
            if isinstance(p, int):
                new_pad.append((p, p))
            elif isinstance(p, tuple) and len(p) == 2:
                new_pad.append(p)
            else:
                break
        if len(new_pad) == rank:
            return new_pad
    raise ValueError(f'Invalid padding format: {padding}, should be str, int,'
                     f' or a sequence of len {rank} where each element is an'
                     f' int or pair of ints.')
                                          
##########-------------------old transformers---------------------############
class GraphChebyshevTransformer_old(nn.Module): 
    """
    Abstract Graph transformer layer. Will break up the inputs super graphs each representing a patch of 
    the original graph, chebyshev transform each patch individually using a padded 
    (to take care of masked vertices), block diagonal graph laplacian.
    will then chebyshev transform along the num_patches dim using a full graph laplacian. afterwards, it will
    act with a convolution of kernel (K_patch, K_super, 4**p).
    inputs: array of shape (N, M, F) where N is the batch dim, M are the number of vertices and F
            are the number of features.
    --------
    :param indices: an array of vertex indices of shape (M, ) where masked vertices are set to -1
    :param N_sup: number of vertices of the super graph, M%N_sup == 0
    :param L_p: graph laplacian for the patches, of block diagonal, padded form
    :param L_s: graph laplacian for the super graph
    :param K_p: number of chebyshev polynomials to use for the "in patch" transformation
    :param K_s: number of chebyshev polynomials to use for the super graph transformation
    :param p: reduction factor (for healpix graphs, might not work as well for other graphs)
    :param Fout: number of output features
    :param use_bias: whether to use bias or not
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    --------
    outputs: array of shape (N, M', Fout) where M' represents the change in the number of vertices due to pooling.
             M' = M//(4**p) for healpix graphs.
    """
    
    N_sup: int
    L_p: Array
    L_s: Array
    K_p: int
    K_s: int
    Fout: int
    p: int
    indices: Array
    #Fout_p: int
    #Fout_s: int
    use_bias: bool = True
    #use_bias_p: bool = True
    #use_bias_s: bool = True
    kernel_init: Callable = nn.initializers.he_normal()
    #p_kernel_init: Callable = nn.initializers.he_normal()
    #s_kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    #p_bias_init: Callable = nn.initializers.zeros
    #s_bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32 
        
    
    def setup(self):
        self.conv = nn.Conv(features=self.Fout, kernel_size=(self.K_p, self.K_s, 4**self.p),
                            strides=(self.K_p, self.K_s, 4**self.p), padding='VALID',
                            use_bias=self.use_bias, dtype=self.dtype, param_dtype=self.param_dtype,
                            kernel_init=self.kernel_init, bias_init=self.bias_init)
        #conv2...
    
    def __call__(self, inputs: Array) -> Array:
        inputs = jnp.asarray(inputs, self.dtype)  #input shape (B, M, F)
        M, F = inputs.shape[-2], inputs.shape[-1] #get input shape
        x = v_init_array(self.K_p, inputs) #shape (B, K_p, M, F)
        #chebyshev transform the patches
        x = v_chebyshev_transform(self.K_p, self.L_p, x) #shape (B, K_p, M, F)
        x = jnp.transpose(x, axes=(0,2,3,1)) #shape (B, N_sup*(M/N_sup), F, K_p)
        x = jnp.reshape(x, (-1, self.N_sup, M//self.N_sup, F, self.K_p)) #shape (B, N_sup, M/N_sup, F, K_p)
        x = jnp.reshape(x, (-1, self.N_sup, (M//self.N_sup)*F*self.K_p)) #shape (B, N_sup, M/N_sup * F*K_p)
        
        x = v_init_array(self.K_s, x) #shape (B, K_s, N_sup, M/N_sup * F*K_p)
        #chebyshev transform the super_pix
        x = v_chebyshev_transform(self.K_s, self.L_s, x) #shape (B, K_s, N_sup, M/N_sup * F*K_p)
        x = jnp.reshape(x, (-1, self.K_s, self.N_sup, (M//self.N_sup)*F, self.K_p)) #shape (B, K_s, N_sup, M/N_sup * F, K_p)
        x = jnp.transpose(x, axes=(0,4,1,2,3)) #shape (B, K_p, K_s, N_sup, M/N_sup *F)
        x = jnp.reshape(x, (-1, self.K_p, self.K_s, self.N_sup, (M//self.N_sup), F)) #(B, K_p, K_s, N_s, M/N, F)
        x = jnp.reshape(x, (-1, self.K_p, self.K_s, M, F)) #(B, K_p, K_s, M, F)
        
        #we want the input to be of valid pixels only. but the indices will be padded.
        #the padding of the maps should be handled by the wrapper. 
        #just pass the unpadded indices after graph laplacian calculation is done.
        #we then mask(unpad) the maps before acting with a conv.
        #unmasked_indices = indices[indices >= 0] #get rid of -1s
        #unmasked_indices = _reduce_indices(unmasked_indices, self.p) #get the reduced indices for pooling
        x = jnp.take(x, self.indices, axis=-2) #(B, K_p, K_s, M', F) #apply the mask
        x = self.conv(x)  #(B, (1), (1), M'/(4**p), Fout) #do the conv
        x = jnp.squeeze(x, axis=(-3, -4)) #shape (N, M'/(4**p), Fout) #squeeze the (1) dims
        print(x.shape)
        #no need to pad.
        return x
    
    

    
class HealpyGraphTransformer_old(nn.Module):
    """
    Healpy wrapper for the abstract Graph transformer layer. Will break up the input maps into patches 
    of super pixels, chebyshev transform each patch individually using a padded 
    (to take care of masked vertices), block diagonal graph laplacian.
    will then chebyshev transform along the num_patches dim using a full graph laplacian. afterwards, it will
    act with a convolution of kernel (K_patch, K_super, 4**p).
    inputs: either a pytree of the form {'nside': nside_in, 'indices': indices_in, 'maps': array of shape (N, M, F)}
            or an array of (N, M, F). If the input is just an array, parameters nside and indices 
            need to be specified. for the transformer, indices_in have to be padded, i.e indices[masked_pixels] = -1.
    --------
    :param N_s: nside of the super pixel map
    :param K_p: number of chebyshev polynomials to use for the "in patch" transformation
    :param K_s: number of chebyshev polynomials to use for the super graph transformation
    :param n_neighbors_p: n_neighbors for the in-patch graph laplacian
    :param n_neighbors_s: n_neighbors for the super pixel map   
    :param p: reduction factor (for healpix graphs, might not work as well for other graphs)
    :param Fout: number of output features
    :param use_bias: whether to use bias or not
    :param nside: optional int, nside of the input maps. needs to be specified if inputs are not of the pytree 
                  form specified above.
    :param indices: optional array, valid indices of the input maps. needs to be specified if inputs are not of 
                    the pytree form specified above. 
    :param kernel_init: kernel initializer, defaults to 'he_normal'
    :param bias_init: bias initializer, defaults to 'zeros'
    :param dtype: dtype of the inputs
    :param param_dtype: dtype of the kernel
    --------
    returns: output dict of form {'nside': nside_out, 'indices': indices_out, 'maps': (N, M', Fout)}
             where M' = M/4**p, nside_out = nside_in/2**p and indices_out are valid indices after a reduction
             in nside by a factor of p
    """
    nside_sup: int
    K_p: int
    K_s: int
    Fout: int
    n_neighbors_p: int
    n_neighbors_s: int
    p: int = 0
    use_bias: bool = True
    nside: Optional[int] = None
    indices: Optional[Array] = None
    kernel_init: Callable = nn.initializers.he_normal()
    bias_init: Callable = nn.initializers.zeros
    dtype: Dtype = jnp.float32
    param_dtype: Dtype = jnp.float32     
        
    @nn.nowrap
    def _get_layer(self, indices_in, L_p, L_s):
        return GraphChebyshevTransformerSingleConv(N_sup=12*(self.nside_sup**2), L_p=L_p, L_s=L_s,
                                                   K_p=self.K_p, K_s=self.K_s, Fout=self.Fout, p=self.p,
                                                   indices=indices_in, 
                                                   use_bias=self.use_bias, 
                                                   kernel_init=self.kernel_init,
                                                   bias_init=self.bias_init, dtype=self.dtype, 
                                                   param_dtype=self.param_dtype)
    
    @nn.compact
    def __call__(self, inputs):
        if not isinstance(inputs, dict):
            if self.nside == None or self.indices == None:
                raise ValueError("If the inputs are maps only, arguments nside and indices need to be specified.")
            else:
                nside_in = self.nside
                indices_in = self.indices
                x = inputs           
        elif isinstance(inputs, dict):
            nside_in, indices_in, x = inputs['nside'], inputs['indices'], inputs['maps'] #padded indices are not jittable
                                                                                         #indices are unpadded
        nside_out = nside_in//(2**self.p)
        
        
        #get new indices
        reduced_indices, indices_out = _get_new_indices(indices=indices_in, p=self.p, reduce=True)
        #calculate padded indices:
        base_padded = -jnp.ones(12*nside_in**2)
        padded_reduced_indices = base_padded.at[reduced_indices].set(reduced_indices)
        #prepare padded input to the graph layer:
        graph_in = jnp.empty((x.shape[0], 12*nside_in**2, x.shape[2]))
        #scatter the input maps onto the empty array
        graph_in = graph_in.at[...,reduced_indices,:].set(x)        
        #get the graph laplacians
        base_L = get_L(nside=nside_in, indices=indices_in, n_neighbors=self.n_neighbors_p)
        L_p= block_padded_L(L=base_L, nside=nside_in, nside_sup=self.nside_sup, indices=indices_in)
        L_s = get_L(nside=self.nside_sup, indices=np.arange(12*(self.nside_sup**2)), n_neighbors=self.n_neighbors_s)
        
        #get the graph layer:
        graphtransformer = self._get_layer(indices_in=reduced_indices, 
                                           L_p=L_p, L_s=L_s)
        #act with the transformer:
        graph_out = graphtransformer(graph_in)
        #get rid of the padding:
        out = graph_out[...,indices_out,:]
        #x = jnp.take(graph_out, indices_out, axis=-2) #(B, K_p, K_s, M', F) #apply the mask
        #x = self.conv(x)  #(B, (1), (1), M'/(4**p), Fout) #do the conv
        #x = jnp.squeeze(x, axis=(-3, -4)) #shape (N, M'/(4**p), Fout) #squeeze the (1) dims
        #construct the output dict:
        outputs = {}
        outputs['nside'] = nside_out
        outputs['indices'] = indices_out
        outputs['maps'] = out
        
        return outputs
        
    
    
    
###########-------------------------index manipulation tools----------------------------------------########
@partial(jit, static_argnums=1)
def _get_boolean_mask(indices, p: int): #indices_to_use = indices[_get_boolean_mask(indices, p)==1]
    """
    Minimally reduce indices to ensure that it can be safely pooled from nside -> nside/(2**p)
    -------------
    :param indices: array of pixel indices in NEST ordering
    :param p: reduction factor, nside -> nside/(2**p)
    -------------
    returns: array of pixel indices in NEST ordering 
    """
    indices_2 = indices - indices % (4**p)
    indices_2 = indices_2 - jnp.roll(indices_2, -(4**p)+1)
    indices_2 = jnp.where(indices_2 == 0, indices, 0)
    mask = jnp.where(indices_2 == 0, 0, 1)
    mask = jax.lax.fori_loop(0, 2**p, lambda i, x: x+jnp.roll(x, 2**i),mask)
    return mask


def _reduce_indices(indices, p): #maybe turn this into a jittable func
    """
    Minimally reduce indices to ensure that it can be safely pooled from nside -> nside/(2**p)
    -------------
    :param indices: array of pixel indices in NEST ordering
    :param p: reduction factor, nside -> nside/(2**p)
    -------------
    returns: array of pixel indices in NEST ordering 
    """
    indices_2 = indices - indices % (4**p)
    indices_2 = indices_2 - np.roll(indices_2, -(4**p)+1)
    new_indices = np.repeat(indices[indices_2==0],4**p)
    new_indices = np.reshape(new_indices, (-1, 4**p))
    new_indices = new_indices + np.arange((4**p))
    new_indices = new_indices.flatten()
    return new_indices

def _transform_indices(indices, p, reduce=True):
    """
    Calculates the new set of map indices after a change in nside
    --------------
    :param indices: array of pixel indices in NEST ordering that are valid for reduction if reduce=True
    :param p: int >= 0, reduction or boost factor
    :param reduce: bool, if True, nside -> nside/(2**p) if false nside -> nside*(2**p)
    --------------
    returns: array of pixel indices in NEST ordering.
    """
    if reduce:
        super_pix = indices%(4**p)
        new_indices = indices[super_pix==0]//(4**p)
    else:
        new_indices = np.zeros(shape=(4**p)*indices.shape[0],dtype=np.int64)
        new_indices = np.repeat((4**p)*indices, (4**p))
        new_indices = new_indices + np.arange((4**p)*indices.shape[0])%(4**p)
    return new_indices

def _get_new_indices(indices, p, reduce=True):
    """
    Helper function to calculate the reduced indices to use in masking and the new indices to pass it along to
    the next module.
    ---------------
    :param indices: array. can be padded with -1s for masked pixels or can just be the valid pixel indices
    :param p: int>=0. reduction(boost) factor for reduce=True(False)
    :param reduce: whether we are reducing nside or boosting it by a factor of 2**p
    ---------------
    returns: reduced_indices, new_indices
             reduced indices: the minimal set of indices that can be reduced by a factor of p
                              to be used in masking before pooling/convolving
             new_indices: the new set of map indices after a change in nside. will be padded if the input is 
                          padded and will be valid pixels only if the input is valid pixels only.
    """
    
    def _reduce_indices(indices, p): #maybe turn this into a jittable func
        """
        Minimally reduce indices to ensure that it can be safely pooled from nside -> nside/(2**p)
        -------------
        :param indices: array of pixel indices in NEST ordering
        :param p: reduction factor, nside -> nside/(2**p)
        -------------
        returns: array of pixel indices in NEST ordering 
        """
        indices_2 = indices - indices % (4**p)
        indices_2 = indices_2 - np.roll(indices_2, -(4**p)+1)
        new_indices = np.repeat(indices[indices_2==0],4**p)
        new_indices = np.reshape(new_indices, (-1, 4**p))
        new_indices = new_indices + np.arange((4**p))
        new_indices = new_indices.flatten()
        return new_indices

    def _transform_indices(indices, p, reduce=True):
        """
        Calculates the new set of map indices after a change in nside
        --------------
        :param indices: array of pixel indices in NEST ordering that are valid for reduction if reduce=True
        :param p: int >= 0, reduction or boost factor
        :param reduce: bool, if True, nside -> nside/(2**p) if false nside -> nside*(2**p)
        --------------
        returns: array of pixel indices in NEST ordering.
        """
        if reduce == True:
            super_pix = indices%(4**p)
            new_indices = indices[super_pix==0]//(4**p)
        elif reduce == False:
            new_indices = np.zeros(shape=(4**p)*indices.shape[0],dtype=np.int64)
            new_indices = np.repeat((4**p)*indices, (4**p))
            new_indices = new_indices + np.arange((4**p)*indices.shape[0])%(4**p)
        return new_indices    
    
    
    unmasked_indices = indices[indices>=0] #get the unmasked indices if indices are padded
    reduced_indices = _reduce_indices(indices=unmasked_indices, p=p) #calculate the reduced indices for pooling
    #if p=0, new_indices == reduced_indices == unmasked_indices, we just need to return reduced_indices and new_indices
    new_indices = _transform_indices(indices=reduced_indices, p=p, reduce=reduce) #calculate the new indices after pool/unpool
    if np.amin(indices)<0: #find out if the indices are 'padded' or not.
        if reduce == True:
            output = -jnp.ones(indices.shape[0]//(4**p)) #create array of -1s of s
            new_indices = output.at[new_indices].set(new_indices) #scatter new indices in
        elif reduce == False:
            output = -jnp.ones(indices.shape[0]*(4**p)) #create array of -1s of s
            new_indices = output.at[new_indices].set(new_indices) #scatter new indices in
    return reduced_indices, new_indices  #reduced_indices for masking, new_indices for passing it
                                         #to the next module.
    
def _get_indices_idx(indices, reduced_indices):
    """
    Helper function to get indices of reduced map indices in a bigger index array
    ------------
    :param indices: 1D `ndarray` of map indices
    :param reduced_indices: 1D `ndarray` of map indices, `len(reduced_indices) <= len(indices)`
    ------------
    returns: 1D `ndarray` of indices corresponding to locations where `indices == reduced_indices`
    """
    idx = np.searchsorted(indices, reduced_indices)
    mask = idx < indices.size
    mask[mask] = indices[idx[mask]] == reduced_indices[mask]
    idx = idx[mask]
    return idx
