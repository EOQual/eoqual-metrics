#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
A simple implementation of the Mean Structural Similarity (MSSIM) algorithm as described in the paper: 
Zhou Wang et al. "Image quality assessment: From error visibility to structural similarity". 
In: Image Processing, IEEE Transactions on 13.4 (2004), pp. 600–612.

Usage: ./mssim.py REFERENCE_IMAGE OTHER_IMAGES...

https://github.com/4og/mssim
"""
import sys, os
import numpy as np
import imageio
from scipy.ndimage import filters

class MSSIM:
    def gaussian(self, size, sigma):
        x = np.arange(0, size, 1, float)
        y = x[:,np.newaxis]
        xc = (size - 1) / 2
        yc = (size - 1) / 2

        gauss = np.exp(-((x - xc)**2 + (y - yc)**2) / (2 * sigma**2))
        return gauss / gauss.sum()

    def compute_from_file(self, fn, fns, k=[0.01, 0.03]):
        c1 = (k[0] * 255)**2
        c2 = (k[1] * 255)**2
        win = self.gaussian(11, 1.5)

        im1 = imageio.imread(fn)
        mu1 = filters.correlate(im1, win)
        mu1_sq = mu1 * mu1
        s1sq =filters.correlate(im1*im1, win) - mu1_sq

        for f in fns:
            im2 = imageio.imread(f)
            if im1.shape != im2.shape:
                print("{}: Incorrect image. All images "
                      "should be of equal size".format(f))
                continue

            mu2 = filters.correlate(im2, win)
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2

            s2sq = filters.correlate(im2*im2, win) - mu2_sq
            s12 = filters.correlate(im1*im2, win) - mu1_mu2

            ssims = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / \
                    ((mu1_sq + mu2_sq + c1) * (s1sq + s2sq + c2))
            
            print("{:24} {:.4f}".format(os.path.basename(f), ssims.mean()))


    def compute(self, GT, P, k=[0.01, 0.03]):
        c1 = (k[0] * 255)**2
        c2 = (k[1] * 255)**2
        win = self.gaussian(11, 1.5)

        mu1 = filters.correlate(GT, win)
        mu1_sq = mu1 * mu1
        s1sq =filters.correlate(GT*GT, win) - mu1_sq

        mu2 = filters.correlate(P, win)
        mu2_sq = mu2 * mu2
        mu1_mu2 = mu1 * mu2

        s2sq = filters.correlate(P*P, win )- mu2_sq
        s12 = filters.correlate(P*P, win) - mu1_mu2

        ssims = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / \
                ((mu1_sq + mu2_sq + c1) * (s1sq + s2sq + c2))
        
        return (ssims.mean())


if __name__ == '__main__':

    if len(sys.argv) < 3:
        print("Usage: mssim.py reference-image other-images ...")
        exit()

    MSSIM().compute_from_file(sys.argv[1], sys.argv[2:])