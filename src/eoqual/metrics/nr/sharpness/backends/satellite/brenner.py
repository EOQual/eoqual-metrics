"""
Mesure de netteté Brenner.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def brenner(image: npt.NDArray) -> float:
    """
    Calcule la netteté d'une image par la mesure de Brenner.

    ``score = Σ (I[x+2, y] − I[x, y])²`` : le calcul saute un pixel en x afin
    de détecter des variations abruptes (contours forts). Rapide et sensible
    aux contours forts, mais peu robuste au bruit.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.

    Returns
    -------
    float
        Score de Brenner (valeur positive, sans unité).
        Valeur plus élevée = image plus nette.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = brenner(img)
    """
    img = image.astype(np.float64)
    diff = img[2:, :] - img[:-2, :]
    return float(np.sum(diff ** 2))
