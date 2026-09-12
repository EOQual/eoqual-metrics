"""
Mesure de netteté Brenner (direction verticale uniquement).

Référence
---------
Brenner, J. F., et al. (1976). *An automated microscope for cytologic
research: A preliminary evaluation*. The Journal of Histochemistry and
Cytochemistry.

Ne pas confondre avec ``fmeasure(algo="BREN")`` (voir
:mod:`eoqual.metrics.nr.focus_fmeasure`) : cette implémentation reprend la
définition originale de 1976 (une seule direction), alors que ``BREN``
implémente l'extension bidirectionnelle de Santos et al. (1997)
(``max(dh, dv)²``, moyenne au lieu de somme). Les deux scores ne sont donc
pas comparables entre eux.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt


def brenner_vertical(image: npt.NDArray) -> float:
    """
    Calcule la netteté d'une image par la mesure de Brenner (verticale).

    ``score = Σ (I[x+2, y] − I[x, y])²`` : le calcul saute un pixel en x afin
    de détecter des variations abruptes (contours forts). Rapide et sensible
    aux contours forts, mais peu robuste au bruit. Une seule direction
    (verticale) — voir la note de module pour la variante bidirectionnelle.

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
    >>> score = brenner_vertical(img)
    """
    img = image.astype(np.float64)
    diff = img[2:, :] - img[:-2, :]
    return float(np.sum(diff ** 2))
