#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SSIM_Index.py : This is an improved implementation of SSIM, based on version of:
    Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr,
    http://isit.u-clermont1.fr/~anvacava

    References:
        [1] Z. Wang, A. C. Bovik, H. R. Sheikh and E. P. Simoncelli.
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13(4):600--612, 2004.

        [2] Z. Wang and A. C. Bovik.
        Mean squared error: Love it or leave it? - A new look at signal fidelity measures.
        IEEE Signal Processing Magazine, 26(1):98--117, 2009.
"""
# Compatibilite Python 3
from __future__ import print_function, division, absolute_import, unicode_literals

__author__ = "Helder C. R. de Oliveira"
__copyright__ = "No Copyright"
__credits__ = ["Helder C. R. de Oliveira"]
__license__ = "GPL"
__version__ = "20171009"
__maintainer__ = "Olivier Amram"
__email__ = "heldercro@gmail.com"
__status__ = "Development" # Prototype / Development / Production
__url__ = 'http://helderc.net, https://github.com/helderc/src/blob/master/SSIM_Index.py'


#-----------------------------------------------------------------------------------------------------------------------------
# IMPORTS STANDARD
#-----------------------------------------------------------------------------------------------------------------------------
import os
import sys

#-----------------------------------------------------------------------------------------------------------------------------
# IMPORTS TIERS
#-----------------------------------------------------------------------------------------------------------------------------
try:
    import numpy as np
    import scipy

except Exception as e :
    print("Probleme d'import de librairie Python : %s" %e)
    print("         (%s)" %os.path.abspath(__file__))
    sys.exit(-2)


def compute_ssim_improved(im1, im2, k=(0.01, 0.03), dynamic_range=255.0, MAP=False):
    # k1,k2 & c1,c2 depend on L (width of color map)
    c_1 = (k[0]*dynamic_range)**2
    c_2 = (k[1]*dynamic_range)**2

    window = np.ones((8, 8))

    # window = gauss_2d((11, 11), 1.5)
    # Normalization
    window /= np.sum(window)

    # Convert image matrices to double precision (like in the Matlab version)
    im1 = im1.astype(float)
    im2 = im2.astype(float)

    # Means obtained by Gaussian filtering of inputs
    mu_1 = scipy.ndimage.filters.convolve(im1, window)
    mu_2 = scipy.ndimage.filters.convolve(im2, window)

    # Squares of means
    mu_1_sq = mu_1**2
    mu_2_sq = mu_2**2
    mu_1_mu_2 = mu_1 * mu_2

    # Squares of input matrices
    im1_sq = im1**2
    im2_sq = im2**2
    im12 = im1*im2

    # Variances obtained by Gaussian filtering of inputs' squares
    sigma_1_sq = scipy.ndimage.filters.convolve(im1_sq, window)
    sigma_2_sq = scipy.ndimage.filters.convolve(im2_sq, window)

    # Covariance
    sigma_12 = scipy.ndimage.filters.convolve(im12, window)

    # Centered squares of variances
    sigma_1_sq -= mu_1_sq
    sigma_2_sq -= mu_2_sq
    sigma_12 -= mu_1_mu_2

    if (c_1 > 0) & (c_2 > 0):
        ssim_map = ((2*mu_1_mu_2 + c_1) * (2*sigma_12 + c_2)) / ((mu_1_sq + mu_2_sq + c_1) * (sigma_1_sq + sigma_2_sq + c_2))
    else:
        numerator1 = 2 * mu_1_mu_2 + c_1
        numerator2 = 2 * sigma_12 + c_2

        denominator1 = mu_1_sq + mu_2_sq + c_1
        denominator2 = sigma_1_sq + sigma_2_sq + c_2

        ssim_map = np.ones(mu_1.size)

        index = (denominator1 * denominator2 > 0)

        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    # return MSSIM
    index = np.mean(ssim_map)

    if MAP:
        return index, ssim_map
    else:
        return index


#def gauss_2d(shape=(3, 3), sigma=0.5):
#    """
#    Code from Stack Overflow's thread
#    2D gaussian mask - should give the same result as MATLAB's
#    fspecial('gaussian',[shape],[sigma])
#    """
#    m, n = [(ss-1.)/2. for ss in shape]
#    y, x = np.ogrid[-m:m+1, -n:n+1]
#    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
#    h[h < np.finfo(h.dtype).eps*h.max()] = 0
#    sumh = h.sum()
#    if sumh != 0:
#        h /= sumh
#    return h

#EOF