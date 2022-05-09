import numpy as np
import matplotlib
import tensorflow as tf
import tensorflow_datasets as tfds
tf.config.set_visible_devices([], 'GPU')
import healpy as hp

from numba import njit, prange, jit, objmode

import math
#from tqdm.auto import tqdm
from tqdm import tqdm
from tqdm.contrib import tenumerate

import jax.numpy as jnp
import jax
from flax import jax_utils

import warnings
import os
import pathlib as path
import glob
import re

import importlib

from src import processtools as pt

class dataset():
    """
    Methods for generating tensorflow datasets. Will first load all the a_lm realizations
    and a mask file (in .fits format) from a relevant directory and calculate relevant map indices
    for a given masking strategy. 
    Almost all the methods in this class are available in processtools.py if they need to be used
    for a specific use case but this class is meant to be a 'one stop shop' to generate the datasets.
    """
    def __init__(self,
                 data_dir,
                 mask_path=None,
                 nside=128,
                 reduction_factor=None,
                 trainperc=0.8,
                 evalperc=0.05):
        """
        data_dir: directory containing the .npy files of a_lm realizations
        mask_path: path to the mask file, in .fits format
        nside: nside of the map data to be created.
        reduction_factor: only needed if the masking strategy is 'aggresive' or 'padded_data'
        trainperc: percentage of dataset to reserve for training
        evalperc: percentage of dataset to reserve for evaluation
                  1-trainperc-evalperc is reserved for testing
        """
        self.data_dir = data_dir
        self.mask_path = mask_path
        self.nside = nside
        self.trainperc = trainperc
        self.evalperc = evalperc
        self.full_pix = np.arange(hp.nside2npix(self.nside))
        if self.mask_path:
            #load the mask and get the unmasked pixels
            print("Loading mask...")
            self.mask = hp.read_map(self.mask_path, nest=False) #we will convert to nest.
            print("Mask loading complete.")
            print("----------------------")
            print("Calculating map indices...")
            self.unmasked_pix, self.masked_pix = self._mask()
            self.adaptive_pix = self._reduce_indices(self.unmasked_pix, p=1)
            if reduction_factor:
                self.reduction_factor = reduction_factor
                self.aggressive_pix = self._reduce_indices(self.unmasked_pix, p=self.reduction_factor)
                self.padded_pix = self._extend_indices(self.unmasked_pix, nside=self.nside, 
                                                       p=self.reduction_factor) 
            if reduction_factor is None:
                warnings.warn("reduction_factor is not specified. pixel indices for 'aggressive',"
                              "and 'padded_data' masking strategies will not be available.")
                
        print("Map indices calculation complete.")
        print("---------------------------------")
        print("Loading data...")
        _alms = []
        _labels = []
        for i, path in enumerate(glob.glob(self.data_dir+'/*.npy')):
            pattern = "L(.*?)l"
            label = "L"+re.search(pattern, path).group(1)
            _labels.append(label[0:-1])
            _alms.append(np.load(path))
        print("Data loading complete")
        print("---------------------")
        self.num_classes = len(_labels)
        print("Number of labels: {}".format(self.num_classes))
        print("Labels are: ", _labels)
        #labels = np.concatenate([np.repeat(np.asarray(_labels[i]),_alms[i].shape[0]) 
        #                               for i in range(len(_labels))])
        labels = np.concatenate([i*np.ones(_alms[i].shape[0]) for i in range(len(_alms))]).astype(np.int8)
        _alms = np.concatenate(tuple(_alms))
        print("-----------------")
        _alms = self._rotate_alm(_alms)
        print("-----------------------")
        
        seed = np.random.randint(1,high=25)
        np.random.RandomState(seed).shuffle(_alms)
        np.random.RandomState(seed).shuffle(labels)
        
        print("Creating training, testing and evaluation splits")

        (self.alms_train, 
         self.alms_test, 
         self.alms_eval) = np.split(_alms, [np.int64(_alms.shape[0]*self.trainperc),
                                                np.int64(_alms.shape[0]*(1-self.evalperc))])
        (self.labels_train, 
         self.labels_test, 
         self.labels_eval) = np.split(labels, [np.int64(labels.shape[0]*self.trainperc),
                                                  np.int64(labels.shape[0]*(1-self.evalperc))])
        
        print("Training split: {}%, testing split: {}%, validation_split: {}%".format(int(100*self.trainperc),
                                                                                      math.ceil(100*(1-self.trainperc
                                                                                      -self.evalperc)),
                                                                                      int(100*self.evalperc)))        
        
        self.train_size = self.alms_train.shape[0]
        self.test_size = self.alms_test.shape[0]
        self.eval_size = self.alms_eval.shape[0]
        self.num_classes = len(_labels),
        self.label_names = _labels
        self.labels = labels
    
    def create_tf_datasets(self, global_batch_size, prefetch):
        """
        Prepares batched and prefetched tensorflow datasets
        -------------
        global_batch_size: global batch size, local patch size (per device) = global/num_device
        prefetch: amount to prefetch
        -------------
        returns train_ds, test_ds as tf.data.Datasets objects
        """
        self.global_batch_size = global_batch_size
        self.prefetch = prefetch
        print("Preparing tf.data.Datasets...")
        train_ds = tf.data.Dataset.from_tensor_slices({"maps": self.maps_train, "labels": self.labels_train})
        test_ds = tf.data.Dataset.from_tensor_slices({"maps": self.maps_test, "labels": self.labels_test})
        train_ds = train_ds.repeat()
        test_ds = test_ds.repeat()
        train_ds = train_ds.shuffle(self.train_size)
        train_ds = train_ds.batch(global_batch_size)
        test_ds = test_ds.batch(global_batch_size)
        train_ds = train_ds.prefetch(prefetch)
        print("Dataset preparation complete.")
        return train_ds, test_ds
    
    def rotate_train_dataset(self):
        """
        Rotates the training dataset randomly.
        """
        print("Rotating a_lms reserved for training...")
        print("---------------------------------------")
        alms_rot = self._rotate_alm(self.alms_train, lmax=250)
        print("Preparing new training maps...")
        maps_train = self._norm(self._get_maps(alms_rot.astype(np.complex128),self.nside))
        train_ds = tf.data.Dataset.from_tensor_slices({"maps": maps_train, "labels": self.labels_train})
        train_ds = train_ds.shuffle(self.train_size)
        train_ds = train_ds.batch(self.global_batch_size)
        train_ds = train_ds.prefetch(self.prefetch)
        print("Dataset rotation complete.")
        return train_ds    
    
    def prepare_maps(self, indices, output_shape):
        """
        Creates healpy maps and saves them as instance attributes
        -----------
        indices: either 'unmasked', 'adaptive', 'aggressive', 'padded', or 'full'
        output_shape: either 'valid_only' or 'full'
        """
        print("Preparing maps...")
        print("Preparing training maps...")
        maps_train = self._norm(self._get_maps(self.alms_train.astype(np.complex128),self.nside))
        print("Preparing test maps...")
        maps_test = self._norm(self._get_maps(self.alms_test.astype(np.complex128),self.nside))
        print("Preparing evaluation maps...")
        maps_eval = self._norm(self._get_maps(self.alms_eval.astype(np.complex128),self.nside))
        
        if indices=='unmasked':
            _indices=self.unmasked_pix
        elif indices=='adaptive':
            _indices=self.adaptive_pix
        elif indices=='aggressive':
            _indices=self.aggressive_pix
        elif indices=='padded':
            _indices=self.unmasked_pix
        elif indices=='full':
            _indices=self.full_pix
                             
        #masking
        x_train2 = []
        x_test2 = []
        x_eval2 = []
        npix = hp.nside2npix(self.nside)
        temp_map = np.zeros(npix)
        for sample in maps_train:
            temp_map[_indices] = sample[_indices]
            if output_shape=='full':
                x_train2.append(temp_map)
            elif output_shape=='valid_only':
                if indices=='padded':
                    x_train2.append(temp_map[self.padded_pix])
                else:
                    x_train2.append(temp_map[_indices])
        x_train2 = np.array(x_train2).astype(np.float16)[...,None]

        temp_map = np.zeros(npix)
        for sample in maps_test:
            temp_map[_indices] = sample[_indices]
            if output_shape=='full':
                x_test2.append(temp_map)
            elif output_shape=='valid_only':
                if indices=='padded':
                    x_test2.append(temp_map[self.padded_pix])
                else:
                    x_test2.append(temp_map[_indices])
        x_test2 = np.array(x_test2).astype(np.float16)[...,None]

        temp_map = np.zeros(npix)
        for sample in maps_eval:
            temp_map[_indices] = sample[_indices]
            if output_shape=='full':
                x_eval2.append(temp_map)
            elif output_shape=='valid_only':
                if indices=='padded':
                    x_eval2.append(temp_map[self.padded_pix])
                else:
                    x_eval2.append(temp_map[_indices])
        x_eval2 = np.array(x_eval2).astype(np.float16)[...,None]
        
        self.maps_train = x_train2
        self.maps_test = x_test2
        self.maps_eval = x_eval2
        self.maps_eval_unmasked = maps_eval
        "Map preparation complete."

    def _reduce_indices(self, indices, p):
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
    
    
    def _extend_indices(self, indices, nside, p):
        """
        Minimally extends a set of indices such that it can be pooled from nside -> nside/2**p
        -------------
        :param indices: array of pixel indices in NEST ordering
        :param nside: nside of the input
        :param p: reduction factor nside -> nside/(2**p)
        :return: array of pixel indices in NEST ordering
        -------------
        """
        # get the map to reduce
        m_in = np.zeros(hp.nside2npix(nside))
        m_in[indices] = 1.0
        nside_out = nside // (2**p)

        # reduce
        m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in="NEST", order_out="NEST")

        # expand
        m_in = hp.ud_grade(map_in=m_in, nside_out=nside, order_in="NEST", order_out="NEST")

        # get the new indices
        return np.arange(hp.nside2npix(nside))[m_in > 1e-12]
    
    
    def _transform_indices(self, indices, p, reduce=True):
        if reduce:
            super_pix = indices%(4**p)
            new_indices = indices[super_pix==0]//4**p
        else:
            new_indices = np.zeros(shape=(4**p)*indices.shape[0],dtype=np.int64)
            new_indices = np.repeat((4**p)*indices, (4**p))
            new_indices = new_indices + np.arange((4**p)*indices.shape[0])%(4**p)
        return new_indices
    
    def _mask(self):
        """
        Utility function to get the unmasked and masked pixels
        """
        mask = hp.ud_grade(self.mask,self.nside,pess=True,order_in='RING',dtype=np.float64)
        for i,pix in enumerate(mask):
            if pix >= 0.5:
                mask[i] = 1
            else:
                mask[i] = 0
        mask = hp.reorder(mask, r2n=True)
        unmasked_pix = np.nonzero(mask>0)[0]
        masked_pix = np.nonzero(mask==0)[0]
        return unmasked_pix, masked_pix
    
    def _rotate_alm(self, alm, lmax=250):
        """
        Function to apply a random rotation to each set of a_lms.
        :param alm: np.array of a_lms of shape=(N, (lmax+1)*(lmax+2)/2) for multiple a_lms or
                    np.array of shape=((lmax+1)*(lmax+2)/2, ) for a_lms of a single map. 
                    a_lms must be ordered in the default healpy scheme
        :param lmax: lmax of the a_lms, assumes every alm has the same lmax

        returns rotated a_lms, output shape is the same as the input shape.
        """
        print("Rotating a_lms...")
        if len(alm.shape)==1:
            ang1, ang2, ang3 = 360*np.random.sample(size=(3,1))
            rot_custom = hp.Rotator(rot=[ang1,ang2,ang3])
            rotalm = rot_custom.rotate_alm(alm, lmax=lmax, inplace=False)
            print("a_lm rotations complete.")
            return np.array(rotalm).astype(np.complex128)
        else:
            ang1, ang2, ang3 = 360*np.random.sample(size=(3,alm.shape[0]))
            rotalms = []
            pbar = tqdm(alm)
            for i, sample in enumerate(pbar):
                rot_custom = hp.Rotator(rot=[ang1[i],ang2[i],ang3[i]])
                rotsample = rot_custom.rotate_alm(sample, lmax=lmax, inplace=False)
                rotalms.append(rotsample)
            print("a_lm rotations complete.")
            return np.array(rotalms).astype(np.complex128)
        
    @jit(forceobj=True)
    def _get_maps(self, a_lm, nside):
        """
        Function to get map realizations from the a_lms.
        ------------
        :param a_lm: np.array of a_lms of shape (N, (lmax+1)*(lmax+2)/2) where N is the number of maps and
                     lmax is the lmax of the a_lms or a single array of shape ((lmax+1)*(lmax+2)/2, ).
                     a_lms must be in the default healpy ordering scheme
        :param nside: nside of the maps
        ------------
        Returns: np.array of healpy maps.
        """
        if len(a_lm.shape)==1:
            return np.array(hp.reorder(hp.alm2map(a_lm, nside, pol=False),
                                       inp='RING',out='NESTED')).astype(np.float32)
        else:
            return np.array([hp.reorder(hp.alm2map(a_lm_i, nside, pol=False),
                                        inp='RING',out='NESTED') for a_lm_i in tqdm(a_lm)]).astype(np.float32)
        
    @jit(forceobj=True) 
    def _norm(self, maps):
        """
        Function to normalize the maps so that their pixel values lie between 0 and 1.
        --------------
        :param maps: np.array of maps of shape (N, npix) or a single array of shape (npix, ) where N is the 
                     number of maps
        --------------
        Returns normalized maps with shape equal to input shape.
        """
        if len(maps.shape)==1:
            minval = np.full_like(maps,np.min(maps))
            rangeval = np.full_like(maps,np.max(maps)-np.min(maps))
            normed_map = (maps-minval)/rangeval
            return normed_map.astype(np.float32)
        else:
            normed_maps = np.full_like(maps,1)
            for i,sample in enumerate(maps):
                minval = np.full_like(sample,np.min(sample))
                rangeval = np.full_like(sample,np.max(sample)-np.min(sample))
                normed_maps[i] = (sample-minval)/rangeval
            return normed_maps.astype(np.float32) 
    
    