"""
Weighted Signal-to-Noise Ratio (WSNR) — pondération par une fonction de
sensibilité au contraste (CSF).

Implémentation propre, écrite à partir de la description publiée du modèle
de CSF de Mannos & Sakrison (voir Référence) — remplace l'implémentation
précédemment vendored (`backends/metrikz/metrikz.py::wsnr`, GPLv2, voir
`THIRD_PARTY_LICENSES.md`), désormais disponible en algo `metrikz` pour
compatibilité ascendante uniquement.

Principe
--------
1. Erreur ``ref - query``, transformée de Fourier 2D, recentrée (DC au centre).
2. Une carte de fréquence radiale est construite par pixel fréquentiel,
   avec une correction angulaire (effet oblique — la sensibilité au
   contraste humaine est légèrement plus faible sur les diagonales que sur
   les axes horizontal/vertical).
3. Cette fréquence est pondérée par le modèle de CSF de Mannos & Sakrison
   (plateau en basse fréquence, décroissance exponentielle au-delà).
4. Le spectre d'erreur est multiplié par la CSF ; le rapport signal/bruit
   pondéré est exprimé en dB.

Référence
---------
Mannos, J. & Sakrison, D. (1974). *The effects of a visual fidelity
criterion on the encoding of images*. IEEE Trans. Information Theory,
20(4), 525-536. Modèle de CSF également repris par de nombreux outils de
qualité d'image (MeTriX MuX, sewar, pymetrikz) sous une forme équivalente.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def _mannos_sakrison_csf(radial_freq: npt.NDArray) -> npt.NDArray:
    """
    Évalue le modèle de CSF de Mannos & Sakrison sur une carte de fréquence
    radiale (cycles/degré).

    Plateau à ``0.9809`` sous le pic de sensibilité (~7.89 c/deg), puis
    décroissance ``2.6 * (0.0192 + 0.114 f) * exp(-(0.114 f)^1.1)``.
    """
    csf = 2.6 * (0.0192 + 0.114 * radial_freq) * np.exp(-((0.114 * radial_freq) ** 1.1))
    csf[radial_freq < 7.8909] = 0.9809
    return csf


def _radial_frequency_map(rows: int, cols: int) -> npt.NDArray:
    """
    Construit une carte de fréquence radiale corrigée de l'effet oblique
    (la CSF humaine décroît un peu plus vite sur les diagonales que sur les
    axes) pour une image de shape ``(rows, cols)``.
    """
    hr, hc = rows / 2.0 - 0.5, cols / 2.0 - 0.5
    y, x = np.mgrid[-hr:hr + 1, -hc:hc + 1]
    # Nombre de cycles représentés sur la largeur de l'image (échelle usuelle
    # du modèle, cf. implémentations de référence des outils cités ci-dessus).
    n_freq = 60.0
    plane = (x + 1j * y) / rows * 2.0 * n_freq
    angle = np.angle(plane)
    w = 0.7
    obliquity = ((1.0 - w) / 2.0) * np.cos(4.0 * angle) + (1.0 + w) / 2.0
    return np.abs(plane) / obliquity


def wsnr(GT: npt.NDArray, P: npt.NDArray) -> float:
    """
    Calcule le Weighted Signal-to-Noise Ratio (WSNR) entre deux images.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence, niveaux de gris ``(H, W)``.
    P : npt.NDArray
        Image traitée / dégradée, même shape.

    Returns
    -------
    float
        WSNR en dB. Valeur plus élevée indique une meilleure qualité
        (l'erreur pondérée perceptuellement est plus faible). ``inf`` si
        les deux images sont identiques (erreur pondérée nulle).

    Examples
    --------
    >>> import numpy as np
    >>> ref = np.random.rand(64, 64)
    >>> score = wsnr(ref, ref)
    >>> score == float("inf")
    True
    """
    ref = GT.astype(np.float64)
    query = P.astype(np.float64)
    rows, cols = ref.shape

    radial_freq = _radial_frequency_map(rows, cols)
    csf = _mannos_sakrison_csf(radial_freq)

    err_spectrum = np.fft.fftshift(np.fft.fft2(ref - query)) * csf
    signal_spectrum = np.fft.fft2(ref)

    weighted_error_power = np.real(np.sum(err_spectrum * np.conj(err_spectrum)))
    signal_power = np.real(np.sum(signal_spectrum * np.conj(signal_spectrum)))

    if weighted_error_power == 0:
        return float("inf")
    return float(10.0 * np.log10(signal_power / weighted_error_power))
