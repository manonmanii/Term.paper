# coding: utf-8
import numpy as np
from .. import img_as_float
from ._denoise_cy import _denoise_bilateral, _denoise_tv_bregman


def denoise_bilateral(image, win_size=5, sigma_range=None, sigma_spatial=1,
                      bins=10000, mode='constant', cval=0):
   
    return _denoise_bilateral(image, win_size, sigma_range, sigma_spatial,
                              bins, mode, cval)


def denoise_tv_bregman(image, weight, max_iter=100, eps=1e-3, isotropic=True):
    
    return _denoise_tv_bregman(image, weight, max_iter, eps, isotropic)


def _denoise_tv_chambolle_3d(im, weight=100, eps=2.e-4, n_iter_max=200):
    

    px = np.zeros_like(im)
    py = np.zeros_like(im)
    pz = np.zeros_like(im)
    gx = np.zeros_like(im)
    gy = np.zeros_like(im)
    gz = np.zeros_like(im)
    d = np.zeros_like(im)
    i = 0
    while i < n_iter_max:
        d = - px - py - pz
        d[1:] += px[:-1]
        d[:, 1:] += py[:, :-1]
        d[:, :, 1:] += pz[:, :, :-1]

        out = im + d
        E = (d ** 2).sum()

        gx[:-1] = np.diff(out, axis=0)
        gy[:, :-1] = np.diff(out, axis=1)
        gz[:, :, :-1] = np.diff(out, axis=2)
        norm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
        E += weight * norm.sum()
        norm *= 0.5 / weight
        norm += 1.
        px -= 1. / 6. * gx
        px /= norm
        py -= 1. / 6. * gy
        py /= norm
        pz -= 1 / 6. * gz
        pz /= norm
        E /= float(im.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


def _denoise_tv_chambolle_2d(im, weight=50, eps=2.e-4, n_iter_max=200):
   

    px = np.zeros_like(im)
    py = np.zeros_like(im)
    gx = np.zeros_like(im)
    gy = np.zeros_like(im)
    d = np.zeros_like(im)
    i = 0
    while i < n_iter_max:
        d = -px - py
        d[1:] += px[:-1]
        d[:, 1:] += py[:, :-1]

        out = im + d
        E = (d ** 2).sum()
        gx[:-1] = np.diff(out, axis=0)
        gy[:, :-1] = np.diff(out, axis=1)
        norm = np.sqrt(gx ** 2 + gy ** 2)
        E += weight * norm.sum()
        norm *= 0.5 / weight
        norm += 1
        px -= 0.25 * gx
        px /= norm
        py -= 0.25 * gy
        py /= norm
        E /= float(im.size)
        if i == 0:
            E_init = E
            E_previous = E
        else:
            if np.abs(E_previous - E) < eps * E_init:
                break
            else:
                E_previous = E
        i += 1
    return out


def denoise_tv_chambolle(im, weight=50, eps=2.e-4, n_iter_max=200,
                         multichannel=False):
    
    im_type = im.dtype
    if not im_type.kind == 'f':
        im = img_as_float(im)

    if im.ndim == 2:
        out = _denoise_tv_chambolle_2d(im, weight, eps, n_iter_max)
    elif im.ndim == 3:
        if multichannel:
            out = np.zeros_like(im)
            for c in range(im.shape[2]):
                out[..., c] = _denoise_tv_chambolle_2d(im[..., c], weight, eps,
                                                       n_iter_max)
        else:
            out = _denoise_tv_chambolle_3d(im, weight, eps, n_iter_max)
    else:
        raise ValueError('only 2-d and 3-d images may be denoised with this '
                         'function')
    return out
