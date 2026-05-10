"""
eoqual_metrics — Collection of image quality metrics.

Public API: import metrics directly from this package.

Examples
--------
>>> import eoqual.metrics as m
>>> score = m.psnr(gt_image, processed_image)
>>> score = m.sharpness(image, algo='tenengrad')

>>> from eoqual.metrics.runner import MetricsRunner
>>> from eoqual.metrics.export.console import display_table
>>> from eoqual.metrics.export.csv import save_csv, load_csv
"""

__version__     = "0.5.0"
__title__       = "eoqual.metrics"
__description__ = "Collection of image quality metrics"
__url__         = ""
__uri__         = __url__
__doc__         = __description__ + " <" + __uri__ + ">"
__author__      = "Olivier Amram"
__email__       = "olivier.amram@free.fr"
__license__     = "MIT"
__copyright__   = "Copyright (c) 2026 Olivier Amram"

from .config import METRICS_CONFIGS  # noqa: F401

# FR metrics
from .fr.error import mae, mdae, mse, nrmse, rmse, rmse_sw  # noqa: F401
from .fr.signal import snr, psnr, psnrb, psnrc, wsnr  # noqa: F401
from .fr.ssim import ssim, msssim, cw_ssim  # noqa: F401
from .fr.correlation import ncc, ndp, nmi, scc, uqi  # noqa: F401
from .fr.perceptual import (  # noqa: F401
    dists, fsim, gmsd, issm, lpips_vgg, mad, nlpd, pamse, reco, vif, vsi,
)
from .fr.spectral import ergas, rase, sam, sre  # noqa: F401

# NR metrics
from .nr.brisque import brisque  # noqa: F401
from .nr.niqe import niqe  # noqa: F401
from .nr.piqe import piqe  # noqa: F401
from .nr.focus import entropy, fmeasure  # noqa: F401
from .nr.sharpness import sharpness  # noqa: F401
from .nr.sharpness.backends.gradient.tenengrad import tenengrad  # noqa: F401
from .nr.sharpness.backends.gradient.laplacian import laplacian  # noqa: F401
from .nr.sharpness.backends.gradient.sobel import sobel_sharpness  # noqa: F401
from .nr.sharpness.backends.phase.lpc_si import lpc_si  # noqa: F401
from .nr.sharpness.backends.phase.psi import psi  # noqa: F401
from .nr.sharpness.backends.phase.sharpness_index import sharpness_index  # noqa: F401
from .nr.sharpness.backends.spectral.s3 import s3  # noqa: F401
from .nr.sharpness.backends.spectral.mtf import mtf  # noqa: F401
from .runner import MetricsRunner  # noqa: F401

__all__ = [
    "METRICS_CONFIGS",
    # FR — error
    "mae", "mdae", "mse", "nrmse", "rmse", "rmse_sw",
    # FR — signal (order matches METRICS_CONFIGS)
    "snr", "psnr", "psnrb", "psnrc", "wsnr",
    # FR — ssim
    "ssim", "msssim", "cw_ssim",
    # FR — correlation
    "ncc", "ndp", "nmi", "scc", "uqi",
    # FR — perceptual
    "dists", "fsim", "gmsd", "issm", "lpips_vgg", "mad", "nlpd",
    "pamse", "reco", "vif", "vsi",
    # FR — spectral
    "ergas", "rase", "sam", "sre",
    # NR — blind IQA
    "brisque", "niqe", "piqe",
    # NR — focus
    "entropy", "fmeasure",
    # NR — sharpness (façade + backends directs)
    "sharpness",
    "tenengrad", "laplacian", "sobel_sharpness",
    "lpc_si", "psi", "sharpness_index",
    "s3", "mtf",
    # Runner
    "MetricsRunner",
]
