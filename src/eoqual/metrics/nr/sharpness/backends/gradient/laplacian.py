"""
Mesure de netteté par variance du Laplacien.

La variance du Laplacien est l'une des mesures de netteté les plus simples
et les plus répandues. Les images nettes présentent des transitions abruptes
d'intensité, ce qui se traduit par un Laplacien de grande amplitude et donc
une variance élevée. Les images floues ont un Laplacien aplati.

Référence
---------
Pech-Pacheco, J. L., et al. *Diatom autofocusing in brightfield microscopy*.
ICPR 2000.
"""
from __future__ import annotations
import numpy as np
import numpy.typing as npt
import cv2


def laplacian(image: npt.NDArray) -> float:
    """
    Calcule la netteté d'une image par la variance du Laplacien.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``, dtype quelconque.
        Les valeurs sont converties en ``float32`` en interne.

    Returns
    -------
    float
        Variance du Laplacien. Valeur plus élevée = image plus nette.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = laplacian(img)
    """
    img = image.astype(np.float32)
    lap = cv2.Laplacian(img, cv2.CV_32F)
    return float(lap.var())
