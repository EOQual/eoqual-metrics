"""
Mesure de netteté par énergie haute fréquence (FFT).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def fft_sharpness(image: npt.NDArray) -> float:
    """
    Calcule la netteté d'une image par ratio d'énergie haute fréquence (FFT).

    1. Transformée de Fourier 2-D, déplacement du zéro au centre.
    2. Masquage d'un disque central (basses fréquences) pour ne conserver
       que les hautes fréquences.
    3. Score = ``Σ|F_HF| / N_HF`` (énergie haute fréquence moyenne).

    Efficace sur les scènes très texturées ; le score augmente avec la
    présence de détails fins.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.

    Returns
    -------
    float
        Ratio d'énergie haute fréquence. Valeur plus élevée = image plus nette.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = fft_sharpness(img)
    """
    img = image.astype(np.float64)
    f_shift = np.fft.fftshift(np.fft.fft2(img))
    magnitude = np.log(np.abs(f_shift) + 1e-10)
    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2
    r = min(rows, cols) // 4  # rayon du disque masqué
    mask = np.ones((rows, cols), dtype=bool)
    mask[crow - r: crow + r, ccol - r: ccol + r] = False
    return float(np.sum(magnitude[mask]) / np.sum(mask))
