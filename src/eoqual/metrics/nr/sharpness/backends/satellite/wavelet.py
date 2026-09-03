"""
Mesure de netteté par énergie pondérée des sous-bandes en ondelettes.
"""
from __future__ import annotations

from typing import List

import numpy as np
import numpy.typing as npt
import pywt

from .....core.image_utils import normalize_histogram


def wavelet(
    image: npt.NDArray,
    wavelet_name: str = "db4",
    level: int = 3,
) -> float:
    """
    Calcule la netteté d'une image par énergie pondérée des sous-bandes
    haute fréquence d'une décomposition en ondelettes.

    * Décomposition dyadique 2-D (``pywt.wavedec2``) jusqu'au niveau ``level``.
    * Pour chaque niveau, énergie logarithmique moyenne de ``cH``, ``cV``, ``cD``.
    * Les énergies sont pondérées par ``2^(level-i-1)`` (les niveaux fins,
      c'est-à-dire les détails les plus fins, comptent davantage).
    * Score final = somme pondérée des énergies.

    Reflète la netteté à plusieurs échelles spatiales simultanément.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    wavelet_name : str, optional
        Nom de l'ondelette (voir ``pywt.wavelist()``). Par défaut ``"db4"``.
    level : int, optional
        Niveau de décomposition. Par défaut ``3``.

    Returns
    -------
    float
        Score de netteté par ondelettes. Valeur plus élevée = image plus
        riche en détails haute fréquence à toutes les échelles.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = wavelet(img)
    """
    img = normalize_histogram(image)
    coeffs = pywt.wavedec2(img, wavelet=wavelet_name, level=level)

    energies: List[float] = []
    weights: List[float] = []
    for i, (cH, cV, cD) in enumerate(coeffs[1:]):
        e_h = np.log10(1.0 + np.mean(cH ** 2))
        e_v = np.log10(1.0 + np.mean(cV ** 2))
        e_d = np.log10(1.0 + np.mean(cD ** 2))
        e_level = 0.25 * (e_h + e_v) + 0.5 * e_d
        energies.append(e_level)
        weights.append(2.0 ** (level - i - 1))

    energies_arr = np.array(energies)
    weights_arr = np.array(weights)
    weights_arr = weights_arr / weights_arr.sum()
    return float(np.sum(energies_arr * weights_arr))
