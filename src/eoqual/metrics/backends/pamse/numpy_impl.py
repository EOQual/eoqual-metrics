#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import ndimage

def pamse(GT, P, rescale=True):
    """
    Compute Perceptual-fidelity Aware Mean Squared Error (PAMSE) IQA metric
    :cite:`xue-2013-perceptual`. This implementation is a translation of the
    reference Matlab implementation provided by the authors of
    :cite:`xue-2013-perceptual`.

    https://github.com/bwohlberg/sporco/blob/master/sporco/metric.py

    Parameters
    ----------
    vref : array_like
      Reference image
    vcmp : array_like
      Comparison image
    rescale : bool, optional (default True)
      Rescale inputs so that `vref` has a maximum value of 255, as assumed
      by reference implementation

    Returns
    -------
    score : float
      PAMSE IQA metric
    """

    # Calculate difference, promoting to float if vref and vcmp have integer
    # dtype
    emap = np.asarray(GT, dtype=np.float64) - np.asarray(P, dtype=np.float64)
    # Input images in reference code on which this implementation is
    # based are assumed to be on range [0,...,255].
    if rescale:
        emap *= (255.0 / GT.max())
    sigma = 0.8
    herr = ndimage.filters.gaussian_filter(emap, sigma)
    score = np.mean(herr**2)
    return score

#EOF