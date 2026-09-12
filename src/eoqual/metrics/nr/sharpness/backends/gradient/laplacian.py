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

Ne pas confondre avec ``fmeasure(algo="LAPV")`` (voir
:mod:`eoqual.metrics.nr.focus_fmeasure`) : même filiation (Pech-Pacheco et
al. 2000, "variance du Laplacien"), mais noyau différent — cette
implémentation utilise le noyau Laplacien 4-connexe par défaut d'OpenCV
(``cv2.Laplacian``, ``[[0,1,0],[1,-4,1],[0,1,0]]``), alors que ``LAPV``
utilise le noyau 8-connexe pondéré ``fspecial('laplacian', 0.2)`` de
MATLAB, hérité de la table de référence Pertuz et al. (2013). Les deux
scores ne sont donc pas comparables entre eux.
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
