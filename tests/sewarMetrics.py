#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sewar is a python package for image quality assessment using different metrics.

Implemented metrics
    Full reference metrics
        Mean Squared Error (MSE)
        Root Mean Sqaured Error (RMSE)
        Peak Signal-to-Noise Ratio (PSNR)
        rRoot Mean Sqaured Error using Sliding Window(rmse_sw)
        Universal Quality Image Index (UQI)
        Structural Similarity Index (SSIM)
        Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS)
        Spatial Correlation Coefficient (SCC)
        Relative Average Spectral Error (RASE)
        Spectral Angle Mapper (SAM)
        Multi-scale Structural Similarity Index (MS-SSIM)
        Visual Information Fidelity (VIF)
        Block Sensitive - Peak Signal-to-Noise Ratio (PSNR-B)
    No reference metrics
        Spectral Distortion Index (D_lambda)
        Spatial Distortion Index (D_S)
        Quality with No Reference (QNR)

https://pypi.org/project/sewar/
https://sewar.readthedocs.io/en/latest/
"""
import sys
import numpy as np
from PIL import Image
from sewar.full_ref import *
# from sewar.full_ref import *

FullReferenceMetrics = ['mse', 'rmse', 'psnr', 'rmse_sw', 'uqi', 'ssim', 'ergas', 'scc', 'rase', 'sam', 'msssim', 'vifp', 'psnrb']
# NoReferenceMetrics = ['d_lambda', 'd_s', 'qnr']

def main():
    argv = sys.argv
    if len(argv) != 3:
        print('usage: python3 -m sewarMetrics.py image1.tif image2.tif', file=sys.stderr)
        sys.exit(2)

    try:
        img1 = np.asarray(Image.open(argv[1]), dtype=np.float32)
        img2 = np.asarray(Image.open(argv[2]), dtype=np.float32)
    except Exception as e:
        e = 'Cannot load images' + str(e)
        print(e, file=sys.stderr)

    MAX = int(np.max(img1) + 0.5)

    print('Full reference metrics')
    print('----------------------')
    for metric in FullReferenceMetrics:
        metricValue = None
        try:
            # Metrique a appliquer
            metricFunction = eval(metric)
            
            if metric in ['psnr', 'ssim', 'msssim']:
                metricValue = metricFunction(img1,img2, MAX=MAX)
            else:
                metricValue = metricFunction(img1,img2)
            if metric in ['rmse_sw', 'ssim']:
                metricValue = metricValue[0]
            if metric in ['msssim']:
                metricValue = metricValue.real
            print("{} = {:.6f}".format(metric, metricValue))
        except Exception as e :
            print("{} : Error".format(metric))
            print(e)

    # print('No reference metrics')
    # print('--------------------')
    # for metric in NoReferenceMetrics:
    #     metricValue = None
    #     try:
    #         # Metrique a appliquer
    #         metricFunction = eval(metric)
    #
    #         metricValue = metricFunction(img2)
    #         print("{} = {:.6f}".format(metric, metricValue))
    #     except Exception as e :
    #         print("{} : Error".format(metric))
    #         print(e)

if __name__ == '__main__':
    sys.exit(main())

#EOF