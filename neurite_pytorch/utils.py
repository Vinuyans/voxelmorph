import itertools
import warnings
import os

import torch
import numpy as np

def interpn(vol, loc, interp_method='linear', fill_value=None):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice 
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'
        fill_value: value to use for points outside the domain. If None, the nearest
            neighbors will be used (default).

    Returns:
        new interpolated volume of the same size as the entries in loc

    If you find this function useful, please cite the original paper this was written for:
        VoxelMorph: A Learning Framework for Deformable Medical Image Registration 
        G. Balakrishnan, A. Zhao, M. R. Sabuncu, J. Guttag, A.V. Dalca. 
        IEEE TMI: Transactions on Medical Imaging. 38(8). pp 1788-1800. 2019. 

        Unsupervised Learning for Probabilistic Diffeomorphic Registration for Images and Surfaces
        A.V. Dalca, G. Balakrishnan, J. Guttag, M.R. Sabuncu. 
        MedIA: Medical Image Analysis. (57). pp 226-236, 2019 

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """

    if isinstance(loc, (list, tuple)):
        # loc = tf.stack(loc, -1)
        loc = torch.stack(loc, -1) # Modified
    nb_dims = loc.shape[-1]
    input_vol_shape = vol.shape

    if len(vol.shape) not in [nb_dims, nb_dims + 1]:
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        # vol = K.expand_dims(vol, -1)
        vol = torch.unsqueeze(vol, -1) # Modified

    # flatten and float location Tensors
    if not torch.is_floating_point(loc): # Modified
        # target_loc_dtype = vol.dtype if vol.dtype.is_floating else 'float32' 
        target_loc_dtype = vol.dtype if torch.is_floating_point(vol) else torch.float32 # Modified
        # loc = tf.cast(loc, target_loc_dtype)
        loc.to(target_loc_dtype) # Modified
    # elif vol.dtype.is_floating and vol.dtype != loc.dtype:
    elif torch.is_floating_point(vol) and vol.dtype != loc.dtype: # Modified
        # loc = tf.cast(loc, vol.dtype)
        loc.to(target_loc_dtype) # Modified

    # if isinstance(vol.shape, (tf.compat.v1.Dimension, tf.TensorShape)):
    if isinstance(vol.shape, (torch.Size)): # Modified
        # volshape = vol.shape.as_list()
        volshape = list(vol.shape) # Modified
    else:
        volshape = vol.shape

    # max_loc = [d - 1 for d in vol.get_shape().as_list()]
    max_loc = [d - 1 for d in list(vol.shape)] # Modified

    # interpolate
    if interp_method == 'linear':
        # floor has to remain floating-point since we will use it in such operation
        # loc0 = tf.floor(loc)
        loc0 = torch.floor(loc) # Modified

        # clip values
        # clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        # loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        clipped_loc = [torch.clamp(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)] # Modified
        loc0lst = [torch.clamp(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)] # Modified
        # get other end of point cube
        # loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        loc1 = [torch.clamp(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)] # Modified
        # locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]
        locs = [[f.to(torch.int32) for f in loc0lst], [f.to(torch.int32) for f in loc1]] # Modified

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        # note reverse ordering since weights are inverse of diff.
        weights_loc = [diff_loc1, diff_loc0]
        checker = torch.clone(diff_loc1[0])

        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0
        for c in cube_pts:

            # get nd values
            # note re: indices above volumes via
            #   https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's
            #   too expensive. Instead we fill the output with zero for the corresponding value.
            #   The CPU version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because gather_nd needs tf.stack which is slow :(
            idx = sub2ind2d(vol.shape[:-1], subs).to(torch.int64)
            # vol_reshape = tf.reshape(vol, [-1, volshape[-1]])
            vol_reshape = torch.reshape(vol, [-1, volshape[-1]]) # Modified
            # vol_val = tf.gather(vol_reshape, idx)            
            vol_val = torch.tensor(vol_reshape[idx]) # Modified

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            
            # tf stacking is slow, we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            # wt = K.expand_dims(wt, -1)
            wt = torch.unsqueeze(wt, -1) # Modified

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest', \
            'method should be linear or nearest, got: %s' % interp_method
        # roundloc = tf.cast(tf.round(loc), 'int32')
        roundloc = torch.round(loc).to(torch.int32) #Modified
        
        # roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        roundloc = [torch.clamp(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)] # Modified

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind2d(vol.shape[:-1], roundloc)
        # interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)
        interp_vol = torch.reshape(vol, [-1, vol.shape[-1]]) # Modified
        interp_vol = torch.tensor(interp_vol[idx]) # Modified
        
    if fill_value is not None:
        out_type = interp_vol.dtype
        # fill_value = tf.constant(fill_value, dtype=out_type)
        # below = [tf.less(loc[..., d], 0) for d in range(nb_dims)]
        # above = [tf.greater(loc[..., d], max_loc[d]) for d in range(nb_dims)]
        # out_of_bounds = tf.reduce_any(tf.stack(below + above, axis=-1), axis=-1, keepdims=True)
        # interp_vol *= tf.cast(tf.logical_not(out_of_bounds), out_type)
        # interp_vol += tf.cast(out_of_bounds, out_type) * fill_value

        fill_value = torch.tensor(fill_value, dtype=out_type) # Modified
        below = [torch.less(loc[..., d], 0) for d in range(nb_dims)] # Modified
        above = [torch.greater(loc[..., d], max_loc[d]) for d in range(nb_dims)] # Modified
        out_of_bounds = torch.any(torch.stack(below + above, dim=-1), dim=-1, keepdim=True) # Modified
        interp_vol *= torch.logical_not(out_of_bounds).to(out_type) # Modified
        interp_vol += out_of_bounds.to(out_type) * fill_value # Modified

    # if only inputted volume without channels C, then return only that channel
    if len(input_vol_shape) == nb_dims:
        assert interp_vol.shape[-1] == 1, 'Something went wrong with interpn channels'
        interp_vol = interp_vol[..., 0]

    return interp_vol

def resize(vol, zoom_factor, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of 
        length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    If you find this function useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

    """

    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)

    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims

    # Avoid unnecessary work.
    if all(z == 1 for z in zoom_factor):
        return vol

    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.as_list()

    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    lin = [torch.linspace(0., vol_shape[d] - 1., new_shape[d]) for d in range(ndims)]
    grid = ndgrid(*lin)

    return interpn(vol, grid, interp_method=interp_method)
  
def ndgrid(*args, **kwargs):
    """
    broadcast Tensors on an N-D grid with ij indexing
    uses meshgrid with ij indexing

    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)

    Returns:
        A list of Tensors

    """
    return meshgrid(*args, indexing='ij', **kwargs)

def meshgrid(*args, **kwargs):
    """

    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    improves runtime by changing the last step to tiling instead of multiplication.
    https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/ops/array_ops.py#L1921

    Broadcasts parameters for evaluation on an N-D grid.
    Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
    of N-D coordinate arrays for evaluating expressions on an N-D grid.
    Notes:
    `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
    When the `indexing` argument is set to 'xy' (the default), the broadcasting
    instructions for the first two dimensions are swapped.
    Examples:
    Calling `X, Y = meshgrid(x, y)` with the tensors
    ```python
    x = [1, 2, 3]
    y = [4, 5, 6]
    X, Y = meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[4, 4, 4],
    #      [5, 5, 5],
    #      [6, 6, 6]]
    ```
    Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
    Returns:
    outputs: A list of N `Tensor`s with rank N.
    Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
    """
    indexing = kwargs.pop("indexing", "xy")
    # name = kwargs.pop("name", "meshgrid")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(torch.reshape(torch.stack(list(x)), (s0[:i] + (-1,) + s0[i + 1::]))) # Modified
    # Create parameters for broadcasting each tensor to the full size
    # TODO Check here
    shapes = [x.size() for x in args]
    sz = [list(x.size())[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype
    if indexing == "xy" and ndim > 1:
        output[0] = torch.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = torch.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow.
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # TODO(nolivia): improve performance with a broadcast
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):
        stack_sz = [*sz[:i], 1, *sz[(i + 1):]]
        if indexing == 'xy' and ndim > 1 and i < 2:
            stack_sz[0], stack_sz[1] = stack_sz[1], stack_sz[0]
        output[i] = torch.tile(output[i], stack_sz)
    return output

def sub2ind2d(siz, subs, **kwargs):
    """
    assumes column-order major
    """
    # subs is a list
    assert len(siz) == len(subs), \
        'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx

def prod_n(lst):
    """
    Alternative to tf.stacking and prod, since tf.stacking can be slow
    """
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod

def gaussian_kernel(sigma,
                    windowsize=None,
                    indexing='ij',
                    separate=False,
                    random=False,
                    min_sigma=0,
                    dtype=torch.float32,
                    seed=None):
    '''
    Construct an N-dimensional Gaussian kernel.

    Parameters:
        sigma: Standard deviations, scalar or list of N scalars.
        windowsize: Extent of the kernel in each dimension, scalar or list.
        indexing: Whether the grid is constructed with 'ij' or 'xy' indexing.
            Ignored if the kernel is separated.
        separate: Whether the kernel is returned as N separate 1D filters.
        random: Whether each standard deviation is uniformily sampled from the
            interval [min_sigma, sigma).
        min_sigma: Lower bound of the standard deviation, only considered for
            random sampling.
        dtype: Data type of the output. Should be floating-point.
        seed: Integer for reproducible randomization. It is possible that this parameter only
            has an effect if the function is wrapped in a Lambda layer.

    Returns:
        ND Gaussian kernel where N is the number of input sigmas. If separated,
        a list of 1D kernels will be returned.

    For more information see:
        https://github.com/adalca/mivt/blob/master/src/gaussFilt.m

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    # Data type.
    # dtype = tf.dtypes.as_dtype(dtype)
    # assert dtype.is_floating, f'{dtype.name} is not a real floating-point type'

    # No pytorch method to convert Pytorch dtype to Numpy dtype
    # TODO add support for all dtypes
    np_dtype = None
    if dtype == torch.float32:
        np_dtype = np.float32
    # Kernel width.
    if not isinstance(sigma, (list, tuple)):
        sigma = [sigma]
    if not isinstance(min_sigma, (list, tuple)):
        min_sigma = [min_sigma] * len(sigma)
    sigma = [max(f, np.finfo(np_dtype).eps) for f in sigma]
    min_sigma = [max(f, np.finfo(np_dtype).eps) for f in min_sigma]

    # Kernel size.
    if windowsize is None:
        windowsize = [np.round(f * 3) * 2 + 1 for f in sigma]
    if not isinstance(windowsize, (list, tuple)):
        windowsize = [windowsize]
    if len(sigma) != len(windowsize):
        raise ValueError(f'sigma {sigma} and width {windowsize} differ in length')

    # Precompute grid.
    center = [(w - 1) / 2 for w in windowsize]
    mesh = [np.arange(w) - c for w, c in zip(windowsize, center)]
    mesh = [-0.5 * x**2 for x in mesh]
    if not separate:
        mesh = np.meshgrid(*mesh, indexing=indexing)
    # mesh = [tf.constant(m, dtype=dtype) for m in mesh]
    mesh = [torch.tensor(m, dtype=dtype) for m in mesh] # Modified

    # Exponents.
    if random:
        seeds = np.random.default_rng(seed).integers(np.iinfo(int).max, size=len(sigma))
        max_sigma = sigma
        sigma = []
        for a, b, s in zip(min_sigma, max_sigma, seeds):
            # sigma.append(tf.random.uniform(shape=(1,), minval=a, maxval=b, seed=s, dtype=dtype))
            sigma.append(torch.FloatTensor(size=(1,)).uniform_(a,b))# Modified
            
    exponent = [m / s**2 for m, s in zip(mesh, sigma)]

    # Kernel.
    if not separate:
        # exponent = [tf.reduce_sum(tf.stack(exponent), axis=0)]
        exponent = [torch.sum(torch.stack(exponent), axis=0)] # Modified
    # kernel = [tf.exp(x) for x in exponent]
    kernel = [torch.exp(x) for x in exponent] # Modifed
    # kernel = [x / tf.reduce_sum(x) for x in kernel]
    kernel = [x / torch.sum(x) for x in kernel] # Modified

    return kernel if len(kernel) > 1 else kernel[0]

def separable_conv(x,
                   kernels,
                   axis=None,
                   batched=False,
                   padding='SAME',
                   strides=None,
                   dilations=None):
    '''
    Efficiently apply 1D kernels along axes of a tensor with a trailing feature
    dimension. The same filters will be applied across features.

    Inputs:
        x: Input tensor with trailing feature dimension.
        kernels: A single kernel or a list of kernels, as tensors or NumPy arrays.
            If a single kernel is passed, it will be applied along all specified axes.
        axis: Spatial axes along which to apply the kernels, starting from zero.
            A value of None means all spatial axes.
        padding: Whether padding is to be used, either "VALID" or "SAME".
        strides: Optional output stride as a scalar, list or NumPy array. If several
            values are passed, these will be applied to the specified axes, in order.
        dilations: Optional filter dilation rate as a scalar, list or NumPy array. If several
            values are passed, these will be applied to the specified axes, in order.

    Returns:
        Tensor with the same type as the input.

    If you find this function useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    '''
    # Shape.
    if not batched:
        x = torch.unsqueeze(x, dim=0) # Modified
    shape_space = x.size()[1:-1] # Modified
    num_dim = len(x.shape[1:-1])

    # Axes.
    if np.isscalar(axis):
        axis = [axis]
    axes_space = range(num_dim)
    if axis is None:
        axis = axes_space
    assert all(ax in axes_space for ax in axis), 'non-spatial axis passed'

    # Conform strides and dilations.
    ones = np.ones(num_dim, np.int32)
    f = map(lambda x: 1 if x is None else x, (strides, dilations))
    f = map(np.ravel, f)
    f = map(np.ndarray.tolist, f)
    f = map(lambda x: x * len(axis) if len(x) == 1 else x, f)
    f = map(lambda x: [(*ones[:ax], x[i], *ones[ax + 1:]) for i, ax in enumerate(axis)], f)
    strides, dilations = f
    assert len(strides) == len(axis), 'number of strides and axes differ'
    assert len(dilations) == len(axis), 'number of dilations and axes differ'

    # Conform kernels.
    if not isinstance(kernels, (tuple, list)):
        kernels = [kernels]
    if len(kernels) == 1:
        kernels = kernels.copy() * len(axis)
    assert len(kernels) == len(axis), 'number of kernels and axes differ'

    # Merge features and batches.
    ind = np.arange(num_dim + 2)
    forward = (0, ind[-1], *ind[1:-1])
    backward = (0, *ind[2:], 1)
    x = torch.permute(x, forward) # Modified
    shape_bc = torch.tensor(x.size()[:2]) # Modified
    shape_bc = torch.reshape(shape_bc, (2,))
    
    import pdb
    # pdb.set_trace()
    # TODO is torch.cat necessary ?
    # x = torch.reshape(x, shape=torch.cat((
    #     torch.reshape(torch.prod(shape_bc), (1,)),
    #     torch.tensor(shape_space),
    #     torch.tensor([1]),
    # ), dim=0).size())
    x = torch.reshape(x, shape=(
        1,
        torch.reshape(torch.prod(shape_bc), (1,)),
        *list(shape_space),
    ))
    # Convolve.
    for ax, k, s, d in zip(axis, kernels, strides, dilations):
        width = np.prod(k.shape)
        # k = tf.reshape(k, shape=(*ones[:ax], width, *ones[ax + 1:], 1, 1))
        # x = tf.nn.convolution(x, k, padding=padding, strides=s, dilations=d)
        # k = torch.reshape(k, shape=(*ones[:ax], width, *ones[ax + 1:], 1, 1)) # Modified
        k_size = [1] * 5
        k_size[ax] = width
        k = torch.reshape(k, shape=k_size) # Modified
        # TODO CHECK ALL OF THIS
        x = torch.nn.functional.conv3d(x, k, stride=s, dilation=d, padding="same") # Modified TODO check


    # Restore dimensions.
    # x = tf.reshape(x, shape=tf.concat((shape_bc, tf.shape(x)[1:-1]), axis=0))
    # x = tf.transpose(x, backward)
    # TODO check tensor shape
    # x = torch.reshape(x, shape=(*list(shape_bc), *list(torch.tensor(x.size()[1:-1])))) # Modified
    x = torch.permute(x, backward) # Modified

    return x if batched else x[0, ...]

def minmax_norm(x, axis=None):
    """
    Min-max normalize tensor using a safe division.

    Arguments:
        x: Tensor to be normalized.
        axis: Dimensions to reduce during normalization. If None, all axes will be considered,
            treating the input as a single image. To normalize batches or features independently,
            exclude the respective dimensions.

    Returns:
        Normalized tensor.
    """
    # x_min = tf.reduce_min(x, axis=axis, keepdims=True)
    # x_max = tf.reduce_max(x, axis=axis, keepdims=True)
    # return tf.compat.v1.div_no_nan(x - x_min, x_max - x_min)

    # Modified
    x_min = torch.reshape(torch.min(x), shape=([1] * len(x.shape))) 
    x_max = torch.reshape(torch.max(x), shape=([1] * len(x.shape)))
    result = None
    if x_max - x_min == 0:
        result = 0
    else:
        result = torch.div(x - x_min, x_max - x_min)
    return result

