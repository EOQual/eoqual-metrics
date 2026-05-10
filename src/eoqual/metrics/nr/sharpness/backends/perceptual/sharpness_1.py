import numpy as np
import numpy.typing as npt
from dom import DOM
import cpbd
from scipy import ndimage


__all__ = [
    "sharpness",
]

MEASURES_SHARPNESS = [
    "cpbd",
    "sharpness_1",
    "pydom",
]


def _sharpness_1(P: npt.NDArray) -> float: # type: ignore
    '''
    Code based on paper:
    A Nonlocal Maximum Likelihood Estimation Method for Rician Noise Reduction
    in MR Images. IEEE Trans. on Med. Imaging, VOL. 28, NO. 2, feb 2009.
    by Lili He and Ian R. Greenshields
    '''
    [gy, gx] = np.gradient(P)

    hx = np.array([[0, 0, 0], [-1, 0, 1], [0, 0, 0]])
    hy = np.array([[0, -1, 0], [0, 0, 0], [0, 1, 0]])

    wx = ndimage.convolve(P, hx)
    wy = ndimage.convolve(P, hy)

    sharp = ((wx ** 2) * (gx ** 2)) + ((wy ** 2) * (gy ** 2))  # type: ignore

    # At that paper the Sharpness is calculated as 'sharp' is. However, it is a
    # huge number. Appliyng  log as above, will reduce that number and then
    # the sharpness will be given in dB.
    sharp_db = 10 * np.log10(sharp.sum())

    return sharp_db 


def sharpness(P: npt.NDArray, algo: str='cpbd') -> float: # type: ignore
    """
    Computes the sharpness metric, no-reference image quality score
   
    Parameters
    ----------
    P : npt.NDArray
        input image
    algo: str
        1. cpbd # https://pypi.org/project/cpbd/
        2. sharpness_1
        3. pydom # https://github.com/umang-singhal/pydom

    Returns
    -------
    float
        sharpness value
    """
    if algo == 'cpbd':
        P255 = P / np.max(P) * 255
        return cpbd.compute(P255)
    # algos/sharpness_1.py
    elif algo == 'sharpness_1':
        return _sharpness_1(P) # type: ignore
    elif algo == 'pydom':
        iqa = DOM()
        return iqa.get_sharpness(P) # type: ignore
    else:
        raise ValueError(f"'algo' value [{algo}] is not correct")

#EOF