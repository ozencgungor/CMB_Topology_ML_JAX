import matplotlib.pyplot as plt
import healpy as hp
import numpy as np
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

from numba import njit, prange, jit, objmode



@jit(forceobj=True)
def get_maps(a_lm, nside):
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
                                    inp='RING',out='NESTED') for a_lm_i in a_lm]).astype(np.float32)

@jit    
def norm(maps):
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

def rotate_alm(alm,lmax):
    """
    Function to apply a random rotation to each set of a_lms.
    :param alm: np.array of a_lms of shape=(N, (lmax+1)*(lmax+2)/2) for multiple a_lms or
                np.array of shape=((lmax+1)*(lmax+2)/2, ) for a_lms of a single map. 
                a_lms must be ordered in the default healpy scheme
    :param lmax: lmax of the a_lms, assumes every alm has the same lmax
    
    returns rotated a_lms, output shape is the same as the input shape.
    """
    
    if len(alm.shape)==1:
        ang1, ang2, ang3 = 360*np.random.sample(size=(3,1))
        rot_custom = hp.Rotator(rot=[ang1,ang2,ang3])
        rotalm = rot_custom.rotate_alm(alm, lmax=lmax, inplace=False)
        return np.array(rotalm).astype(np.complex128)
    else:
        ang1, ang2, ang3 = 360*np.random.sample(size=(3,alm.shape[0]))
        rotalms = []
        for i, sample in enumerate(alm):
            rot_custom = hp.Rotator(rot=[ang1[i],ang2[i],ang3[i]])
            rotsample = rot_custom.rotate_alm(sample, lmax=250, inplace=False)
            rotalms.append(rotsample)
        return np.array(rotalms).astype(np.complex128)

def extend_indices(nside_in, nside_out, indices_in):
    """
    Minimally extends a set of indices such that it can be reduced to nside_out in a healpy fashion, always 
    four pixels reduce naturally to a higher order pixel. Indices must be in NEST ordering.
    :param indices: 1d array of integer pixel ids in NEST ordering.
    :param nside_in: nside of the input.
    :param nside_out: nside of the output.
    :return: returns a set of indices in NEST ordering.
    """

    ordering = "NEST"

    # get the map to reduce
    m_in = np.zeros(hp.nside2npix(nside_in))
    m_in[indices_in] = 1.0

    # reduce
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in=ordering, order_out=ordering)

    # expand
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_in, order_in=ordering, order_out=ordering)

    # get the new indices
    return np.arange(hp.nside2npix(nside_in))[m_in > 1e-12]

def reduce_indices(nside_in, nside_out, indices_in):
    """
    Minimally reduces a set of indices such that it can be reduced to nside_out in a healpy fashion, always 
    four pixels reduce naturally to a higher order pixel. Indices must be in NEST ordering.
    :param indices: 1d array of integer pixel ids
    :param nside_in: nside of the input
    :param nside_out: nside of the output
    :param nest: indices are ordered in the "NEST" ordering scheme
    :return: returns a set of indices in the same ordering as the input.
    """
    
    ordering = "NEST"

    # get the map to reduce
    m_in = np.ones(hp.nside2npix(nside_in))
    m_in[indices_in] = 0.0

    # reduce
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_out, order_in=ordering, order_out=ordering)

    # expand
    m_in = hp.ud_grade(map_in=m_in, nside_out=nside_in, order_in=ordering, order_out=ordering)

    # get the new indices
    return np.arange(hp.nside2npix(nside_in))[m_in < 1e-12]

def get_indices(mask, nside, target_nside):
    """
    Returns the array of indices that will remain unmasked when nside is downgraded to target nside.
    The masking strategy is aggressive as in this function will "pad" and extend the mask such 
    that the array of indices returned will be pixels that will never be pooled or convolved together 
    with the masked pixels as the maps are reduced to target_nside.
    -------------
    :param mask: boolean mask where masked pixels are 0, unmasked pixels are 1. If the mask nside
                 is higher(lower) than the nside of the maps, the mask will be up(down)graded to
                 the map nside first. any pixel value >= 0.5 -> 1 and 0 otherwise after up(down)grading.
    :param nside: nside of the maps to be masked. the mask will be (up)downgraded to this nside
    :param target_nside: final nside of the maps after it's pooled/convolved.
    -------------
    """
    mask = hp.ud_grade(mask,nside,pess=True,order_in='RING',dtype=np.float64)
    for i,pix in enumerate(mask):
        if pix >= 0.5:
            mask[i] = 1
        else:
            mask[i] = 0
    mask = hp.reorder(mask, r2n=True)
    masked_pix = np.nonzero(mask==0)[0]
    extended_masked_pix = extend_indices(nside_in=nside, nside_out=target_nside, indices_in=masked_pix)
    extended_mask = np.ones(hp.nside2npix(nside=nside))
    extended_mask[extended_masked_pix] = 0
    relevant_pix = np.nonzero(extended_mask)[0]
    return relevant_pix
    
def create_dataset(*alm, 
                   relevant_pix, 
                   global_batch_size, 
                   trainperc = 0.75, 
                   evalperc = 0.05,
                   strategy):
    """
    Function to create tf.datasets from the given a_lms. Will save the a_lms and their labels reserved for 
    training for faster rotations of training data between epochs. 
    ---------------
    :param alm: arrays of a_lms of shape (N, (lmax+1)*(lmax+2)/2) where N is the number of realizations
                in each class. 
    :param relevant_pix: array of unmasked pixel indices in NEST ordering.
    :param global_batch_size: number of batches to batch the data into.
    :param trainperc: float, between 0 and 1. the percentage of data to be reserved for training.
    :param evalperc: float, between 0 and 1. the percentage of data to be reserved for evaluation after
                     training. the network will never train on these maps. maps that are not in the 
                     training or evaluation sets will be used as the test dataset.
    :param strategy: a tf.distribute strategy. 
    --------------
    Returns train_dataset: a tf.dataset instance to be used for training, 
            test_dataset: a tf.dataset instance to be used for testing, 
            x_eval, y_eval: maps reserved for evaluation, and their labels,
            x_alm_train, y_train: a_lms reserved for training and their labels, to be used for rotating the
                                  training data between augmentation epochs, if desired.
    """
    x_alm = np.concatenate([alm_ for alm_ in alm]).astype(np.complex128)
    #logits:
    y_full = np.concatenate([i*np.ones(alm[i].shape[0]) for i in range(len(alm))]).astype(np.int8)
    
    seed = np.random.randint(1,high=25)
    np.random.RandomState(seed).shuffle(x_alm)
    np.random.RandomState(seed).shuffle(y_full)
    
    x_alm_train, x_alm_test, x_alm_eval = np.split(x_alm, [np.int64(x_alm.shape[0]*trainperc),
                                                            np.int64(x_alm.shape[0]*(1-evalperc))])
    
    x_alm_train = x_alm_train.astype(np.complex128)
    
    y_train, y_test, y_eval = np.split(y_full, [np.int64(y_full.shape[0]*trainperc),
                                                np.int64(y_full.shape[0]*(1-evalperc))])
    
    x_train = norm(get_maps(rotate_alm(x_alm_train.astype(np.complex128),lmax=250),128))
    x_test = norm(get_maps(rotate_alm(x_alm_test.astype(np.complex128),lmax=250),128))
    x_eval = norm(get_maps(rotate_alm(x_alm_eval.astype(np.complex128),lmax=250),128))
    
    x_train2 = []
    x_test2 = []
    x_eval2 = []
    npix = hp.nside2npix(128)
    temp_map = np.zeros(npix)
    for sample in x_train:
        temp_map[relevant_pix] = sample[relevant_pix]
        x_train2.append(temp_map[relevant_pix])
    x_train2 = np.array(x_train2).astype(np.float16)[...,None]
    
    temp_map = np.zeros(npix)
    for sample in x_test:
        temp_map[relevant_pix] = sample[relevant_pix]
        x_test2.append(temp_map[relevant_pix])
    x_test2 = np.array(x_test2).astype(np.float16)[...,None]
    
    temp_map = np.zeros(npix)
    for sample in x_eval:
        temp_map[relevant_pix] = sample[relevant_pix]
        x_eval2.append(temp_map[relevant_pix])
    x_eval2 = np.array(x_eval2).astype(np.float16)[...,None]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train2, y_train)).shuffle(len(x_train)).batch(global_batch_size) 
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test2, y_test)).batch(global_batch_size) 
    
    if strategy:
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
        test_dataset = strategy.experimental_distribute_dataset(test_dataset)
    
    return train_dataset, test_dataset, x_eval2, y_eval, x_alm_train, y_train

def rotate_train_data(alm, y_train, relevant_pix, global_batch_size, strategy=None):
    """
    Function to apply a rotation on the training a_lms and recreate a tf.dataset from the rotated a_lms.
    ---------
    :param alm: array of a_lms reserved for training, preferably output by createdata()
    :param y_train: array of labels of the training a_lms, preferably output by createdata()
    :param relevant_pix: boolean array of unmasked pixel indices in NEST ordering.
    :param GLOBAL_BATCH_SIZE: number of batches to batch the data into.
    :param strategy: a tf.distribute strategy. If given, will return a tf.distributed.dataset instance.
    ---------
    Returns: train_dataset: a tf.dataset instance.
    """
    x_train = norm(get_maps(rotate_alm(alm,lmax=250),128))
    x_train2 = []
    npix = hp.nside2npix(128)
    temp_map = np.zeros(npix)
    for sample in x_train:
        temp_map[relevant_pix] = sample[relevant_pix]
        x_train2.append(temp_map[relevant_pix])
    x_train2 = np.array(x_train2).astype(np.float16)[...,None]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train2, y_train)).shuffle(len(x_train)).batch(global_batch_size) 
    if strategy:
        train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    
    return train_dataset


#################----------------------masking and rotating maps--------------------------################

def rotate(maps,nside):
    """np array of maps or a single map
        will apply a random rotation to each map
        return shape is same as input
        -assumes every map has the same nside"""
    if len(maps.shape)==1:
        ang1, ang2, ang3 = 360*np.random.sample(size=(3,1))
        rotmap = np.array(sdm.rotate(maps,ang1,ang2,ang3,nside,p=1)).astype(np.float32)
        return rotmap
    else:
        ang1, ang2, ang3 = 360*np.random.sample(size=(3,maps.shape[0]))
        maps = np.array([sdm.rotate(j,ang1[i],ang2[i],ang3[i],nside,p=1) for i,j in enumerate(maps)]).astype(np.float32)
        return maps

def mask(maps,nside,targetnside):
    """numpy array of maps or a single map
       uses deepsphere padded masks, if nside=targetnside, will apply the mask unpadded
       targetnside < nside
       uses the common CMB mask as the base mask
       will set the masked pixels to -2*np.abs(np.min(map))
       assumes everything is in NEST ordering"""
    #masking the maps, masked pixels -> -2*min
    #the usual CMB common mask
    npix = hp.nside2npix(nside=nside)
    mask=hp.read_map('COM_Mask_CMB-common-Mask-Int_2048_R3.fits')
    mask_ud = hp.ud_grade(mask,nside,pess=True)
    for i,pix in enumerate(mask_ud):
        if mask_ud[i] != 1:
            mask_ud[i] = 0
    mask_ud_nest = hp.reorder(mask_ud, r2n=True)
    #create the padded mask
    unmasked_pix = []
    for i,pix in enumerate(mask_ud_nest):
        if mask_ud_nest[i] == 1:
            unmasked_pix.append(i)
    unmasked_pix = np.array(unmasked_pix)        
    padded_pix = utils.extend_indices(unmasked_pix, nside_in=nside, nside_out=targetnside)
    if len(maps.shape)==1:
        temp_map = np.full(npix, 2*np.min(maps))
        temp_map[padded_pix] = maps[padded_pix]
        return temp_map.astype(np.float32)
    else:
        masked_maps = []
        for sample in maps:
            temp_map = np.full(npix, 2*np.min(sample))
            temp_map[padded_pix] = sample[padded_pix]
            masked_maps.append(temp_map)
        masked_maps = np.array(masked_maps).astype(np.float32)
        return masked_maps
    

###########--------------------------------a_lm tools-------------------------------#################    
@jit    
def getid(lmax,l,m):
    return m * (2 * lmax + 1 - m) // 2 + l

@jit
def negalmgen(lmax, alm):
    if len(alm.shape)==1:
        almneg = np.full_like(alm,0)
        for l in range(lmax+1):
            for m in range(l+1):
                almneg[getidx(250,l,m)] = (-1)**m * np.conj(alm[getidx(250,l,m)])
        return almneg.astype(np.complex64)
    else:
        temp = []
        for sample in alm:
            almneg = np.full_like(sample,0)
            for l in range(lmax+1):
                for m in range(l+1):
                    almneg[getidx(250,l,m)] = (-1)**m * np.conj(sample[getidx(250,l,m)])
            temp.append(almneg)
        return np.array(temp).astype(np.complex64)
            
@jit
def fullalmgen(lmax, alm):
    almneg = negalmgen(lmax,alm)
    if len(alm.shape) == 1:
        fullalm = np.full((lmax+1)**2,0.+0.j)
        for l in range(lmax+1):
            for m in range(-l,l+1):
                if m < 0:
                    fullalm[l*(l+1)+m] = almneg[getidx(lmax,l,-m)]
                else:
                    fullalm[l*(l+1)+m] = alm[getidx(lmax,l,m)]
        return fullalm
    else:
        temp=[]
        for i,sample in enumerate(alm):
            fullalm = np.full((lmax+1)**2,0.+0.j)
            for l in range(lmax+1):
                for m in range(-l,l+1):
                    if m < 0:
                        fullalm[l*(l+1)+m] = almneg[i][getidx(lmax,l,-m)]
                    else:
                        fullalm[l*(l+1)+m] = sample[getidx(lmax,l,m)]
            temp.append(fullalm)
        return np.array(temp).astype(np.complex64)

@jit    
def padc_l(lmax, alm):
    padc_l = np.full((lmax+1)**2,0.)
    c_l = hp.alm2cl(alm,lmax=lmax)
    for l in range(lmax+1):
        for m in range(-l,l+1):
            padc_l[l*(l+1)+m] = c_l[l]
    return padc_l

@jit
def C_lmlpmp(lmax, alm):
    fullalmnorm = fullalmgen(lmax,alm)/padc_l(lmax,alm)
    return np.kron(fullalmnorm,np.conj(fullalmnorm))

@jit
def realalmsingle(lmax,alm):
    alm = fullalmgen(lmax, alm)
    if len(alm.shape) == 1:
        for l in range(lmax+1):
            for m in range(-l,l+1):
                if m == 0:
                    alm[l*(l+1)+m] = np.real(alm[l*(l+1)+m])
                if m > 0:
                    if m%2 == 1:
                        alm[l*(l+1)+m] = np.imag(alm[l*(l+1)+m])
                    if m%2 == 0:
                        alm[l*(l+1)+m] = np.real(alm[l*(l+1)+m])
                if m < 0:
                    if m%2 == 0:
                        alm[l*(l+1)+m] = np.imag(alm[l*(l+1)+m])
                    if m%2 == 1:
                        alm[l*(l+1)+m] = np.real(alm[l*(l+1)+m])
        return np.real(alm).astype(np.float32)
@jit
def realalm(lmax,alm):
    return np.array([realalmsingle(lmax,sample) for sample in alm])

