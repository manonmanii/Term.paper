import numpy as np
from skimage.restoration._nl_means_denoising import _nl_means_denoising_2d, \
                    _nl_means_denoising_3d, \
                    _fast_nl_means_denoising_2d, _fast_nl_means_denoising_3d

def nl_means_denoising(image, patch_size=7, patch_distance=11, h=0.1,
                       multichannel=True, fast_mode=True):
   
    if image.ndim == 2:
        image = image[..., np.newaxis]
        multichannel = True
    if image.ndim != 3:
        raise NotImplementedError("Non-local means denoising is only \
        implemented for 2D grayscale and RGB images or 3-D grayscale images.")
    if multichannel:  # 2-D images
        if fast_mode:
            return np.squeeze(np.array(_fast_nl_means_denoising_2d(image,
                                       patch_size, patch_distance, h)))
        else:
            return np.squeeze(np.array(_nl_means_denoising_2d(image,
                                       patch_size, patch_distance, h)))
    else:  # 3-D grayscale
        if fast_mode:
            return np.array(_fast_nl_means_denoising_3d(image, s=patch_size,
                                              d=patch_distance, h=h))
        else:
            return np.array(_nl_means_denoising_3d(image, patch_size,
                                patch_distance, h))

