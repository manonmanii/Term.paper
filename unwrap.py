import numpy as np
import warnings
from six import string_types



def unwrap_phase(image, wrap_around=False, seed=None):
    
    if image.ndim not in (1, 2, 3):
        raise ValueError('Image must be 1, 2, or 3 dimensional')
    if isinstance(wrap_around, bool):
        wrap_around = [wrap_around] * image.ndim
    elif (hasattr(wrap_around, '__getitem__')
          and not isinstance(wrap_around, string_types)):
        if len(wrap_around) != image.ndim:
            raise ValueError('Length of `wrap_around` must equal the '
                             'dimensionality of image')
        wrap_around = [bool(wa) for wa in wrap_around]
    else:
        raise ValueError('`wrap_around` must be a bool or a sequence with '
                         'length equal to the dimensionality of image')
    if image.ndim == 1:
        if np.ma.isMaskedArray(image):
            raise ValueError('1D masked images cannot be unwrapped')
        if wrap_around[0]:
            raise ValueError('`wrap_around` is not supported for 1D images')
    if image.ndim in (2, 3) and 1 in image.shape:
        warnings.warn('Image has a length 1 dimension. Consider using an '
                      'array of lower dimensionality to use a more efficient '
                      'algorithm')

    if np.ma.isMaskedArray(image):
        mask = np.require(np.ma.getmaskarray(image), np.uint8, ['C'])
    else:
        mask = np.zeros_like(image, dtype=np.uint8, order='C')

    image_not_masked = np.asarray(
        np.ma.getdata(image), dtype=np.double, order='C')
    image_unwrapped = np.empty_like(image, dtype=np.double, order='C',
                                    subok=False)

    if image.ndim == 1:
        unwrap_1d(image_not_masked, image_unwrapped)
    elif image.ndim == 2:
        unwrap_2d(image_not_masked, mask, image_unwrapped,
                  wrap_around, seed)
    elif image.ndim == 3:
        unwrap_3d(image_not_masked, mask, image_unwrapped,
                  wrap_around, seed)

    if np.ma.isMaskedArray(image):
        return np.ma.array(image_unwrapped, mask=mask)
    else:
        return image_unwrapped
