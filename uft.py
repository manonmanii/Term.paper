
from __future__ import division, print_function

import numpy as np




def ufftn(inarray, dim=None):
    
    if dim is None:
        dim = inarray.ndim
    outarray = np.fft.fftn(inarray, axes=range(-dim, 0))
    return outarray / np.sqrt(np.prod(inarray.shape[-dim:]))


def uifftn(inarray, dim=None):
    
    if dim is None:
        dim = inarray.ndim
    outarray = np.fft.ifftn(inarray, axes=range(-dim, 0))
    return outarray * np.sqrt(np.prod(inarray.shape[-dim:]))


def urfftn(inarray, dim=None):
   
    if dim is None:
        dim = inarray.ndim
    outarray = np.fft.rfftn(inarray, axes=range(-dim, 0))
    return outarray / np.sqrt(np.prod(inarray.shape[-dim:]))


def uirfftn(inarray, dim=None, shape=None):
    
    if dim is None:
        dim = inarray.ndim
    outarray = np.fft.irfftn(inarray, shape, axes=range(-dim, 0))
    return outarray * np.sqrt(np.prod(outarray.shape[-dim:]))


def ufft2(inarray):
   
    return ufftn(inarray, 2)


def uifft2(inarray):
    
    return uifftn(inarray, 2)


def urfft2(inarray):
    
    return urfftn(inarray, 2)


def uirfft2(inarray, shape=None):
    
    return uirfftn(inarray, 2, shape=shape)


def image_quad_norm(inarray):
    
    if inarray.shape[-1] != inarray.shape[-2]:
        return (2 * np.sum(np.sum(np.abs(inarray) ** 2, axis=-1), axis=-1) -
                np.sum(np.abs(inarray[..., 0]) ** 2, axis=-1))
    else:
        return np.sum(np.sum(np.abs(inarray) ** 2, axis=-1), axis=-1)


def ir2tf(imp_resp, shape, dim=None, is_real=True):
    
    if not dim:
        dim = imp_resp.ndim
   
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    
    for axis, axis_size in enumerate(imp_resp.shape):
        if axis >= imp_resp.ndim - dim:
            irpadded = np.roll(irpadded,
                               shift=-int(np.floor(axis_size / 2)),
                               axis=axis)
    if is_real:
        return np.fft.rfftn(irpadded, axes=range(-dim, 0))
    else:
        return np.fft.fftn(irpadded, axes=range(-dim, 0))


def laplacian(ndim, shape, is_real=True):
    
    impr = np.zeros([3] * ndim)
    for dim in range(ndim):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (ndim - dim - 1))
        impr[idx] = np.array([-1.0,
                              0.0,
                              -1.0]).reshape([-1 if i == dim else 1
                                              for i in range(ndim)])
    impr[([slice(1, 2)] * ndim)] = 2.0 * ndim
    return ir2tf(impr, shape, is_real=is_real), impr
