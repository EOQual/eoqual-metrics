"""
Mesure de netteté par la moyenne absolue du gradient Sobel.

Cette mesure estime la netteté en calculant l'amplitude moyenne des contours
détectés par le filtre de Sobel. Plus les contours sont forts en moyenne,
plus l'image est considérée comme nette.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
from skimage.filters import sobel as _sobel_skimage


def sobel_sharpness(image: npt.NDArray) -> float:
    """
    Calcule la netteté d'une image par la moyenne absolue du gradient Sobel.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
        Les valeurs sont converties en ``float64`` en interne.

    Returns
    -------
    float
        Moyenne absolue des valeurs du filtre Sobel.
        Valeur plus élevée = image plus nette.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = sobel_sharpness(img)
    """
    sob = _sobel_skimage(image.astype(np.float64))
    return float(np.mean(np.abs(sob)))
