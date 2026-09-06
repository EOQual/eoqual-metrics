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
        'algo': [
            {'name': 'numpy'},
            {'name': 'metrikz'},  # legacy, GPLv2 vendored — voir THIRD_PARTY_LICENSES.md
        ],
        'default': 'numpy',
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
            {'name': 'piq'},
            {'name': 'IQA_pytorch', 'only': 'RGB'},  # legacy, voir THIRD_PARTY_LICENSES.md
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
            {'name': 'piq'},  # nécessite des images >= 161x161 (5 échelles)
            {'name': 'IQA_pytorch', 'only': 'RGB'},  # legacy, voir THIRD_PARTY_LICENSES.md
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
    'jsd': {
        'algo': [
            {'name': 'numpy'},
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    # ------------------------------------------------------------------
    # Full-Reference — Perceptual
    # ------------------------------------------------------------------
    'dists': {
        # bascule vers piq (Apache-2.0) — IQA_pytorch conservé en
        # legacy (voir THIRD_PARTY_LICENSES.md).
        'algo': [
            {'name': 'piq'},
            {'name': 'IQA_pytorch', 'only': 'RGB'},
        ],
        'default': 'piq',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'fsim': {
        'algo': [
            {'name': 'image-similarity-measures'},
            {'name': 'piq'},
        ],
        'default': 'image-similarity-measures',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'gmsd': {
        'algo': [
            {'name': 'numpy', 'map': True},
            {'name': 'piq'},
            {'name': 'IQA_pytorch'},  # legacy, voir THIRD_PARTY_LICENSES.md
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
        # relancé via piq (Apache-2.0, backbone VGG16) — cassé
        # depuis longtemps avec IQA_pytorch (non maintenu, incompatible
        # torch >= 1.8.0), conservé en legacy pour mémoire uniquement.
        'algo': [
            {'name': 'piq'},
            {'name': 'IQA_pytorch'},  # legacy, inutilisable en pratique
        ],
        'default': 'piq',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'mad': {
        # Ne fonctionne pas avec torch >= 1.8.0 (limitation de IQA_pytorch,
        # non maintenu) : aucun algo utilisable actuellement. piq n'a pas
        # d'implémentation MAD — aucune alternative permissive identifiée
        # à ce jour (voir THIRD_PARTY_LICENSES.md).
        'algo': [],
        'default': 'IQA_pytorch',
        'metric_mode': 'FR',
        'lower_better': True,
    },
    'nlpd': {
        # piq n'a pas d'implémentation NLPD — reste sur IQA_pytorch (MIT,
        # fonctionnel dans la plage de version bornée par l'extra 'deep',
        # voir pyproject.toml). Aucune alternative permissive identifiée
        # à ce jour (voir THIRD_PARTY_LICENSES.md).
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
            {'name': 'numpy', 'only': 'GRAY'}
        ],
        'default': 'numpy',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'vif': {
        'algo': [
            {'name': 'sewar'},
            {'name': 'metrikz'},
            {'name': 'mscale_impl'},
            {'name': 'wavelet_impl', 'only': 'GRAY'},
            {'name': 'piq'},
#            {'name': 'IQA_pytorch'},   # Ne fonctionne pas avec torch >= 1.8.0
        ],
        'default': 'sewar',
        'metric_mode': 'FR',
        'lower_better': False,
    },
    'vsi': {
        # relancé via piq (Apache-2.0) — cassé depuis longtemps
        # avec IQA_pytorch (non maintenu, incompatible torch >= 1.8.0).
        'algo': [
            {'name': 'piq'},
        ],
        'default': 'piq',
        'metric_mode': 'FR',
        'lower_better': False,  # piq.vsi renvoie l'indice de similarité brut (plus haut = mieux)
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
            {'name': 'numpy'},
            {'name': 'image-quality', 'rgb': True},
            {'name': 'brisque', 'rgb': True},
        ],
        'default': 'numpy',
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'niqe': {
        'algo': [
            {'name': 'numpy'},
        ],
        'default': 'numpy',
        'metric_mode': 'NR',
        'lower_better': True,
    },
    'piqe': {
        'algo': [{'name': 'numpy'}],
        'default': 'numpy',
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
            # Perceptual (external lib)
            {'name': 'cpbd', 'only': 'GRAY'},
            # Satellite (calibré, remote sensing)
            {'name': 'brenner', 'only': 'GRAY'},
            {'name': 'fft', 'only': 'GRAY'},
            {'name': 'antonel', 'only': 'GRAY'},
            {'name': 'blur_kernel', 'only': 'GRAY'},
            {'name': 'wavelet', 'only': 'GRAY'},
            {'name': 'aem', 'only': 'GRAY'},
            {'name': 'sasbem', 'only': 'GRAY'},
        ],
        'default': 'tenengrad',
        'metric_mode': 'NR',
        'lower_better': False,
    },
})
