

from __future__ import division

import numpy as np
import numpy.random as npr
from scipy.signal import convolve2d

from . import uft

__keywords__ = "restoration, image, deconvolution"


def wiener(image, psf, balance, reg=None, is_real=True, clip=True):
   
    if reg is None:
        reg, _ = uft.laplacian(image.ndim, image.shape, is_real=is_real)
    if not np.iscomplexobj(reg):
        reg = uft.ir2tf(reg, image.shape, is_real=is_real)

    if psf.shape != reg.shape:
        trans_func = uft.ir2tf(psf, image.shape, is_real=is_real)
    else:
        trans_func = psf

    wiener_filter = np.conj(trans_func) / (np.abs(trans_func) ** 2 +
                                           balance * np.abs(reg) ** 2)
    if is_real:
        deconv = uft.uirfft2(wiener_filter * uft.urfft2(image),
                             shape=image.shape)
    else:
        deconv = uft.uifft2(wiener_filter * uft.ufft2(image))

    if clip:
        deconv[deconv > 1] = 1
        deconv[deconv < -1] = -1

    return deconv


def unsupervised_wiener(image, psf, reg=None, user_params=None, is_real=True,
                        clip=True):
    
    params = {'threshold': 1e-4, 'max_iter': 200,
              'min_iter': 30, 'burnin': 15, 'callback': None}
    params.update(user_params or {})

    if reg is None:
        reg, _ = uft.laplacian(image.ndim, image.shape, is_real=is_real)
    if not np.iscomplexobj(reg):
        reg = uft.ir2tf(reg, image.shape, is_real=is_real)

    if psf.shape != reg.shape:
        trans_fct = uft.ir2tf(psf, image.shape,  is_real=is_real)
    else:
        trans_fct = psf


    x_postmean = np.zeros(trans_fct.shape)
    
    prev_x_postmean = np.zeros(trans_fct.shape)

   
    delta = np.NAN

    gn_chain, gx_chain = [1], [1]

    
    areg2 = np.abs(reg) ** 2
    atf2 = np.abs(trans_fct) ** 2

    # The Fourier transfrom may change the image.size attribut, so we
    # store it.
    if is_real:
        data_spectrum = uft.urfft2(image.astype(np.float))
    else:
        data_spectrum = uft.ufft2(image.astype(np.float))

    
    for iteration in range(params['max_iter']):
        

        
        precision = gn_chain[-1] * atf2 + gx_chain[-1] * areg2  # Eq. 29
        excursion = np.sqrt(0.5) / np.sqrt(precision) * (
            np.random.standard_normal(data_spectrum.shape) +
            1j * np.random.standard_normal(data_spectrum.shape))

        
        wiener_filter = gn_chain[-1] * np.conj(trans_fct) / precision

        
        x_sample = wiener_filter * data_spectrum + excursion
        if params['callback']:
            params['callback'](x_sample)

        
        gn_chain.append(npr.gamma(image.size / 2,
                                  2 / uft.image_quad_norm(data_spectrum -
                                                          x_sample *
                                                          trans_fct)))


        gx_chain.append(npr.gamma((image.size - 1) / 2,
                                  2 / uft.image_quad_norm(x_sample * reg)))

        
        if iteration > params['burnin']:
            x_postmean = prev_x_postmean + x_sample

        if iteration > (params['burnin'] + 1):
            current = x_postmean / (iteration - params['burnin'])
            previous = prev_x_postmean / (iteration - params['burnin'] - 1)

            delta = np.sum(np.abs(current - previous)) / \
                np.sum(np.abs(x_postmean)) / (iteration - params['burnin'])

        prev_x_postmean = x_postmean

        
        if (iteration > params['min_iter']) and (delta < params['threshold']):
            break

    
    x_postmean = x_postmean / (iteration - params['burnin'])
    if is_real:
        x_postmean = uft.uirfft2(x_postmean, shape=image.shape)
    else:
        x_postmean = uft.uifft2(x_postmean)

    if clip:
        x_postmean[x_postmean > 1] = 1
        x_postmean[x_postmean < -1] = -1

    return (x_postmean, {'noise': gn_chain, 'prior': gx_chain})


def richardson_lucy(image, psf, iterations=50, clip=True):
   
    image = image.astype(np.float)
    psf = psf.astype(np.float)
    im_deconv = 0.5 * np.ones(image.shape)
    psf_mirror = psf[::-1, ::-1]
    for _ in range(iterations):
        relative_blur = image / convolve2d(im_deconv, psf, 'same')
        im_deconv *= convolve2d(relative_blur, psf_mirror, 'same')

    if clip:
        im_deconv[im_deconv > 1] = 1
        im_deconv[im_deconv < -1] = -1

    return im_deconv
