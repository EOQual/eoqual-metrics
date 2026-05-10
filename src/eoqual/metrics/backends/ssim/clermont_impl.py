#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ssim.py : short description

Description...

"""
__author__ = "Antoine Vacavant, ISIT lab, antoine.vacavant@iut.u-clermont1.fr"
__copyright__ = "No Copyright"
__credits__ = ["Antoine Vacavant"]
__license__ = "GPL"
__version__ = "20230301"
__maintainer__ = "Olivier Amram"
__email__ = "olivier.amram@cnes.fr"
__status__ = "Development" # Prototype / Development / Production
__url__ = 'http://isit.u-clermont1.fr/~anvacava/code-ssim-source.html'


#-----------------------------------------------------------------------------------------------------------------------------
# IMPORTS STANDARD
#-----------------------------------------------------------------------------------------------------------------------------
import sys
import os


#-----------------------------------------------------------------------------------------------------------------------------
# IMPORTS TIERS
#-----------------------------------------------------------------------------------------------------------------------------
try:
    import numpy as np
    import scipy
    from numpy.ma.core import exp
    from scipy.constants.constants import pi

except Exception as e :
    print("Probleme d'import de librairie Python : %s" %e)
    print("         (%s)" %os.path.abspath(__file__))
    sys.exit(-2)


'''
The function to compute SSIM
@param param: img_mat_1 1st 2D matrix
@param param: img_mat_2 2nd 2D matrix
'''
def compute_ssim(img_mat_1, img_mat_2, k=(0.01, 0.03), dynamic_range=255.0, MAP=False):
    #Variables for Gaussian kernel definition
    gaussian_kernel_sigma = 1.5
    gaussian_kernel_width = 11
    gaussian_kernel = np.zeros((gaussian_kernel_width, gaussian_kernel_width))
    
    #Fill Gaussian kernel
    for i in range(gaussian_kernel_width):
        for j in range(gaussian_kernel_width):
            gaussian_kernel[i,j] = \
                (1/(2*np.pi*(gaussian_kernel_sigma**2)))* \
                exp(-(((i-5)**2)+((j-5)**2))/(2*(gaussian_kernel_sigma**2)))

    #Convert image matrices to double precision (like in the Matlab version)
    img_mat_1 = img_mat_1.astype(np.float64)
    img_mat_2 = img_mat_2.astype(np.float64)
    
    #Squares of input matrices
    img_mat_1_sq = img_mat_1**2
    img_mat_2_sq = img_mat_2**2
    img_mat_12   = img_mat_1*img_mat_2
    
    #Means obtained by Gaussian filtering of inputs
    img_mat_mu_1 = scipy.ndimage.filters.convolve(img_mat_1, gaussian_kernel)
    img_mat_mu_2 = scipy.ndimage.filters.convolve(img_mat_2 ,gaussian_kernel)
        
    #Squares of means
    img_mat_mu_1_sq = img_mat_mu_1**2
    img_mat_mu_2_sq = img_mat_mu_2**2
    img_mat_mu_12   = img_mat_mu_1*img_mat_mu_2
    
    #Variances obtained by Gaussian filtering of inputs' squares
    img_mat_sigma_1_sq = scipy.ndimage.filters.convolve(img_mat_1_sq, gaussian_kernel)
    img_mat_sigma_2_sq = scipy.ndimage.filters.convolve(img_mat_2_sq, gaussian_kernel)
    
    #Covariance
    img_mat_sigma_12 = scipy.ndimage.filters.convolve(img_mat_12, gaussian_kernel)
    
    #Centered squares of variances
    img_mat_sigma_1_sq = img_mat_sigma_1_sq-img_mat_mu_1_sq
    img_mat_sigma_2_sq = img_mat_sigma_2_sq-img_mat_mu_2_sq
    img_mat_sigma_12   = img_mat_sigma_12-img_mat_mu_12
    
    #c1/c2 constants
    #First use: manual fitting
    c_1 = 6.5025
    c_2 = 58.5225
    
    #Second use: change k1,k2 & c1,c2 depend on dynamic_range
    dynamic_range = 255.0
    c_1 = (k[0]*dynamic_range)**2
    c_2 = (k[1]*dynamic_range)**2
    
    #Numerator of SSIM
    num_ssim = (2*img_mat_mu_12+c_1)*(2*img_mat_sigma_12+c_2)
    #Denominator of SSIM
    den_ssim = (img_mat_mu_1_sq+img_mat_mu_2_sq+c_1)*\
    (img_mat_sigma_1_sq+img_mat_sigma_2_sq+c_2)
    #SSIM
    ssim_map = num_ssim/den_ssim
    index    = np.average(ssim_map)

    if MAP:
        return index, ssim_map
    else:
        return index
