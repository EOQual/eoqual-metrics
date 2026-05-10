#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module providing functionality to implement Structural Similarity Image 
Quality Assessment. Based on original paper by Z. Whang
"Image Quality Assessment: From Error Visibility to Structural Similarity" IEEE
Transactions on Image Processing Vol. 13. No. 4. April 2004.

https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py
https://github.com/RenYang-home/HLVC/blob/master/my_ssim.py
"""
import sys
import numpy as np
import pylab
from PIL import Image
from scipy import signal
from scipy import ndimage


def fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def ssim(img1, img2, cs_map=False):
    """Return the Structural Similarity Map corresponding to input images img1 
    and img2 (images are assumed to be uint8)
    
    This function attempts to mimic precisely the functionality of ssim.m a 
    MATLAB provided by the author's of SSIM
    https://ece.uwaterloo.ca/~z70wang/research/ssim/ssim_index.m

    https://www.cns.nyu.edu/~lcv/ssim/
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)
    K1 = 0.01
    K2 = 0.03
    L = 255 #bitdepth of image
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = signal.fftconvolve(window, img1, mode='valid')
    mu2 = signal.fftconvolve(window, img2, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(window, img1*img1, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(window, img2*img2, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(window, img1*img2, mode='valid') - mu1_mu2
    if cs_map:
        return (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        return ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))


def msssim(img1, img2):
    """This function implements Multi-Scale Structural Similarity (MSSSIM) Image 
    Quality Assessment according to Z. Wang's "Multi-scale structural similarity 
    for image quality assessment" Invited Paper, IEEE Asilomar Conference on 
    Signals, Systems and Computers, Nov. 2003 
    
    Author's MATLAB implementation:-
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip
    """
    level = 5
    weight = np.array([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    downsample_filter = np.ones((2, 2))/4.0
    im1 = img1.astype(np.float64)
    im2 = img2.astype(np.float64)
    mssim = np.array([])
    mcs = np.array([])
    for l in range(level):
        ssim_map, cs_map = ssim(im1, im2, cs_map=True)
        mssim = np.append(mssim, ssim_map.mean())
        mcs = np.append(mcs, cs_map.mean())
        filtered_im1 = ndimage.filters.convolve(im1, downsample_filter, 
                                                mode='reflect')
        filtered_im2 = ndimage.filters.convolve(im2, downsample_filter, 
                                                mode='reflect')
        im1 = filtered_im1[::2, ::2]
        im2 = filtered_im2[::2, ::2]
    return (np.prod(mcs[0:level-1]**weight[0:level-1])*
                    (mssim[level-1]**weight[level-1]))

def main():
    """Compute the SSIM index on two input images specified on the cmd line."""
    argv = sys.argv
    if len(argv) != 3:
        print('usage: python3 ssim_mssim.py image1.tif image2.tif', file=sys.stderr)
        sys.exit(2)

    try:
        img1 = np.asarray(Image.open(argv[1]))
        img2 = np.asarray(Image.open(argv[2]))
    except Exception as e:
        e = 'Cannot load images' + str(e)
        print(e, file=sys.stderr)

    ssim_map = ssim(img1, img2)
    ms_ssim = msssim(img1, img2)
    print('SSIM = {:.6f}  -  MSSIM = {:.6f}'.format(ssim_map.mean(), ms_ssim))

    pylab.figure()
    pylab.subplot(131)
    pylab.title('Image1')
    pylab.imshow(img1, interpolation='nearest', cmap=pylab.gray())
    pylab.subplot(132)
    pylab.title('Image2')
    pylab.imshow(img2, interpolation='nearest', cmap=pylab.gray())
    pylab.subplot(133)
    pylab.title('SSIM Map\n SSIM: %f\n MSSSIM: %f' % (ssim_map.mean(), ms_ssim))
    pylab.imshow(ssim_map, interpolation='nearest', cmap=pylab.gray())
    pylab.tight_layout()
    pylab.show()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

#EOF