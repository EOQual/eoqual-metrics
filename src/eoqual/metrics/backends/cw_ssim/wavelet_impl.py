#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from pywt import cwt

def cw_ssim(GT, P, width=30):
    """
    Compute the complex wavelet SSIM (CW-SSIM)

    https://github.com/charparr/tundra-snow/blob/master/continuous_patterns.py

    Parameters
    ----------
    GT : npt.NDArray
        first (original) input image
    P : npt.NDArray
        second (deformed) input image
    width : int
        width for the wavelet convolution (default: 30)
    
    Returns
    -------
    float
        cw_ssim value
    npt.NDArray
        cw_ssim map
    """
    rows, cols = GT.shape

    # Define a width for the wavelet convolution
    widths = np.arange(1, width+1)

    # Use the image data as arrays
    sig1 = GT.ravel()
    sig2 = P.ravel()

    # Convolution
    cwtmatr1, _ = cwt(sig1, widths, 'mexh')
    cwtmatr2, _ = cwt(sig2, widths, 'mexh')

    # Compute the first term
    c1c2 = np.multiply(abs(cwtmatr1), abs(cwtmatr2))
    c1_2 = np.square(abs(cwtmatr1))
    c2_2 = np.square(abs(cwtmatr2))
    num_ssim_1 = 2 * np.sum(c1c2, axis=0) + 0.01
    den_ssim_1 = np.sum(c1_2, axis=0) + np.sum(c2_2, axis=0) + 0.01

    # Compute the second term
    c1c2_conj = np.multiply(cwtmatr1, np.conjugate(cwtmatr2))
    num_ssim_2 = 2 * np.abs(np.sum(c1c2_conj, axis=0)) + 0.01
    den_ssim_2 = 2 * np.sum(np.abs(c1c2_conj), axis=0) + 0.01

    # Construct the result
    cw_ssim_map = (num_ssim_1 / den_ssim_1) * (num_ssim_2 / den_ssim_2)
    cw_ssim_map = cw_ssim_map.reshape(rows, cols)

    # Average the per pixel results
    cw_ssim_vals = round( np.average(cw_ssim_map), 2) 

    return cw_ssim_vals, cw_ssim_map

#EOF