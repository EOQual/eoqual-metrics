"""
Registre central des métriques disponibles dans eoqual_metrics.

Chaque entrée décrit les algorithmes disponibles pour une métrique donnée,
l'algorithme par défaut, le mode (FR = Full-Reference, NR = No-Reference)
et le sens d'optimalité (``lower_better``).
"""

from collections import OrderedDict
from .nr.focus_fmeasure import MEASURES_FOCUS

METRICS_CONFIGS: OrderedDict = OrderedDict({
    # ------------------------------------------------------------------
    # Full-Reference — Error-based
    # ------------------------------------------------------------------
    'mae': {
        'algo': [
            {'name': 'numpy'}
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'mdae': {
        'algo': [
            {'name': 'numpy'}
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'mse': {
        'algo': [
            {'name': 'numpy'},
            {'name': 'skimage'},
            {'name': 'metrikz'},
            {'name': 'sewar'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'nrmse': {
        'algo': [
            {'name': 'skimage'}
        ],
        'default': 'skimage',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'rmse': {
        'algo': [
            {'name': 'numpy'},
            {'name': 'metrikz'},
            {'name': 'sewar'},
            {'name': 'image-similarity-measures'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'rmse_sw': {
        'algo': [
#            {'name': 'sewar', 'map': True}  # Erreur: float() argument must be a string or a real number, not 'tuple'
        ],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    # ------------------------------------------------------------------
    # Full-Reference — Signal-based
    # ------------------------------------------------------------------
    'snr': {
        'algo': [
            {'name': 'numpy'},
            {'name': 'metrikz'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'psnr': {
        'algo': [
            {'name': 'numpy'},
            {'name': 'skimage'},
            {'name': 'metrikz'},
            {'name': 'sewar'},
            {'name': 'image-similarity-measures'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'psnrb': {
        'algo': [{'name': 'sewar'}],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'psnrc': {
        'algo': [
            {'name': 'numpy', 'only': 'RGB'},
            {'name': 'skimage', 'only': 'RGB'},
            {'name': 'metrikz', 'only': 'RGB'},
            {'name': 'sewar', 'only': 'RGB'},
            {'name': 'image-similarity-measures', 'only': 'RGB'}
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'wsnr': {
        'algo': [{'name': 'metrikz'}],
        'default': 'metrikz',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    # ------------------------------------------------------------------
    # Full-Reference — SSIM family
    # ------------------------------------------------------------------
    'ssim': {
        'algo': [
            {'name': 'skimage'},
            {'name': 'metrikz'},
            {'name': 'sewar'},
            {'name': 'image-similarity-measures', 'only': 'GRAY'},
            {'name': 'ssim_1', 'only': 'GRAY'},
            {'name': 'ssim_2', 'only': 'GRAY'},
            {'name': 'ssim_3', 'only': 'GRAY'},
            {'name': 'ssim_4', 'only': 'GRAY'},
            {'name': 'IQA_pytorch', 'only': 'RGB'},
        ],
        'default': 'skimage',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'msssim': {
        'algo': [
            {'name': 'sewar'},
            {'name': 'msssim_1', 'only': 'GRAY'},
            {'name': 'msssim_2', 'only': 'GRAY'},
            {'name': 'IQA_pytorch', 'only': 'RGB'},
        ],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'cw_ssim': {
        'algo': [
            {'name': 'cw_ssim_1', 'map': True, 'only': 'GRAY'},
#            {'name': 'IQA_pytorch', 'only': 'RGB'}
        ],
        'default': 'cw_ssim_1',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    # ------------------------------------------------------------------
    # Full-Reference — Correlation / Statistical
    # ------------------------------------------------------------------
    'ncc': {
        'algo': [
            {'name': 'numpy'},
            {'name': 'scipy'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'ndp': {
        'algo': [
            {'name': 'numpy'},
            {'name': 'scipy_1'},
            {'name': 'scipy_2'},
            {'name': 'sklearn'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'nmi': {
        'algo': [{'name': 'skimage'}],
        'default': 'skimage',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'scc': {
        'algo': [{'name': 'sewar'}],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'uqi': {
        'algo': [
            {'name': 'metrikz'},
            {'name': 'sewar'},
            {'name': 'image-similarity-measures', 'only': 'GRAY'},
        ],
        'default': 'metrikz',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    # ------------------------------------------------------------------
    # Full-Reference — Perceptual
    # ------------------------------------------------------------------
    'dists': {
        'algo': [
            {'name': 'IQA_pytorch', 'only': 'RGB'}
        ],
        'default': 'IQA_pytorch',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'fsim': {
        'algo': [
            {'name': 'image-similarity-measures'}
        ],
        'default': 'image-similarity-measures',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'gmsd': {
        'algo': [
            {'name': 'numpy', 'map': True},
            {'name': 'IQA_pytorch'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'issm': {
        'algo': [
            {'name': 'image-similarity-measures', 'only': 'GRAY'}
        ],
        'default': 'image-similarity-measures',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'lpips_vgg': {
        'algo': [
    #        {'name': 'IQA_pytorch', 'only': 'RGB'}
        ],
        'algo': [],     # Ne fonctionne pas avec torch >= 1.8.0
        'default': 'IQA_pytorch',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'mad': {
        'algo': [
    #        {'name': 'IQA_pytorch', 'only': 'RGB'}
        ],
        'algo': [],     # Ne fonctionne pas avec torch >= 1.8.0
        'default': 'IQA_pytorch',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'nlpd': {
        'algo': [
            {'name': 'IQA_pytorch'}
        ],
        'default': 'IQA_pytorch',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'pamse': {
        'algo': [
            {'name': 'numpy'}
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'reco': {
        'algo': [
            {'name': 'polar_impl', 'only': 'GRAY'}
        ],
        'default': 'polar_impl',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'vif': {
        'algo': [
            {'name': 'sewar'},
            {'name': 'metrikz'},
            {'name': 'mscale_impl'},
            {'name': 'wavelet_impl', 'only': 'GRAY'},
#            {'name': 'IQA_pytorch'},   # Ne fonctionne pas avec torch >= 1.8.0
        ],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'vsi': {
        'algo': [
#            {'name': 'IQA_pytorch', 'only': 'RGB'}
        ],
        'algo': [],     # Ne fonctionne pas avec torch >= 1.8.0
        'default': 'IQA_pytorch',
        'metric_mode': 'FR',
        'lower_better': None,
    },
    # ------------------------------------------------------------------
    # Full-Reference — Spectral / Remote Sensing
    # ------------------------------------------------------------------
    'ergas': {
        'algo': [
            {'name': 'sewar'}
        ],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'rase': {
        'algo': [
            {'name': 'sewar', 'only': 'GRAY'}
        ],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'sam': {
        'algo': [
            {'name': 'image-similarity-measures'}
        ],
        'default': 'image-similarity-measures',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'sre': {
        'algo': [
            {'name': 'image-similarity-measures'}
        ],
        'default': 'image-similarity-measures',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    # ------------------------------------------------------------------
    # No-Reference — Blind IQA
    # ------------------------------------------------------------------
    'brisque': {
        'algo': [
            {'name': 'image-quality', 'rgb': True},
            {'name': 'libsvm_impl'},
            {'name': 'sklearn_impl'},
            {'name': 'brisque', 'rgb': True},
        ],
        'default': 'libsvm_impl',
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'niqe': {
        'algo': [
##            {'name': 'programtalk_impl'}, # Ne fonctionne pas
            {'name': 'eadcat_impl'},
        ],
        'default': 'eadcat_impl',
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'piqe': {
        'algo': [{'name': 'eadcat_impl'}],
        'default': 'eadcat_impl',
        'metric_mode': 'NR',
        'lower_better': True,
    },
    # ------------------------------------------------------------------
    # No-Reference — Focus / Entropy
    # ------------------------------------------------------------------
    'entropy': {
        'algo': [
            {'name': 'skimage'},
            {'name': 'entropy_1'},
            {'name': 'scipy'},
        ],
        'default': 'entropy_1',
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'fmeasure': {
        'algo': [
            {'name': 'ACMO', 'only': 'GRAY'},
            {'name': 'BREN', 'only': 'GRAY'},
            {'name': 'CONT', 'only': 'GRAY'},
            {'name': 'CURV', 'only': 'GRAY'},
            {'name': 'DCTE', 'only': 'GRAY'},
            {'name': 'DCTR', 'only': 'GRAY'},
            {'name': 'GDER', 'only': 'GRAY'},
            {'name': 'GLLV', 'only': 'GRAY'},
            {'name': 'GLVA', 'only': 'GRAY'},
            {'name': 'GLVN', 'only': 'GRAY'},
            {'name': 'GRAE', 'only': 'GRAY'},
            {'name': 'GRAS', 'only': 'GRAY'},
            {'name': 'GRAT', 'only': 'GRAY'},
            {'name': 'HELM', 'only': 'GRAY'}, 
            {'name': 'HISR', 'only': 'GRAY'}, 
            {'name': 'LAPD', 'only': 'GRAY'}, 
            {'name': 'LAPE', 'only': 'GRAY'}, 
            {'name': 'LAPM', 'only': 'GRAY'}, 
            {'name': 'LAPV', 'only': 'GRAY'}, 
            {'name': 'SFIL', 'only': 'GRAY'}, 
            {'name': 'SFRQ', 'only': 'GRAY'}, 
            {'name': 'TENG', 'only': 'GRAY'}, 
            {'name': 'TENV', 'only': 'GRAY'}, 
            {'name': 'VOLA', 'only': 'GRAY'}, 
            {'name': 'WAVR', 'only': 'GRAY'}, 
            {'name': 'WAVS', 'only': 'GRAY'}, 
            {'name': 'WAVV', 'only': 'GRAY'}
        ],
        'default': 'ACMO',
        'metric_mode': 'NR',
        'lower_better': True,
    },
    # ------------------------------------------------------------------
    # No-Reference — Sharpness
    # ------------------------------------------------------------------
    'sharpness': {
        'algo': [
            # Gradient-based
            {'name': 'tenengrad'},
            {'name': 'laplacian'},
            {'name': 'sobel'},
            # Phase-based
            {'name': 'lpc_si', 'only': 'GRAY'},
            {'name': 'sharpness_index', 'only': 'GRAY'},
            {'name': 'psi', 'only': 'GRAY'},
            # Spectral
            {'name': 's3'},
            {'name': 'mtf', 'only': 'GRAY'},
            # Perceptual (external libs)
            {'name': 'cpbd', 'only': 'GRAY'},
            {'name': 'sharpness_1', 'only': 'GRAY'},
##            {'name': 'pydom'},    # Erreur : unsupported operand type(s) for *: 'int' and 'NoneType'
        ],
        'default': 'tenengrad',
        'metric_mode': 'NR',
        'lower_better': False,
    },
})
