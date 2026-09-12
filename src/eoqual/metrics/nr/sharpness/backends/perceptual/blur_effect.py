"""
Mesure de netteté "Blur Effect" (Crété, Dolmière, Ladret & Nicolas, 2007).

Référence
---------
F. Crété, T. Dolmiere, P. Ladret, M. Nicolas (2007). "The Blur Effect:
Perception and Estimation with a New No-Reference Perceptual Blur
Metric". Proc. SPIE 6492, Human Vision and Electronic Imaging XII,
64920I. https://hal.archives-ouvertes.fr/hal-00232709

Principe (éq. 1-6 du papier) : reflouter l'image avec un filtre moyenneur
1-D et comparer, séparément par axe, la variation entre pixels voisins
avant/après ce reflou — une forte baisse signale une image initialement
nette, une baisse faible une image déjà floue. Score final = maximum sur
les axes (un flou directionnel, ex. bougé, reste détecté même si l'autre
axe est net).

Implémentation : délègue à ``skimage.measure.blur_effect`` plutôt que de
réimplémenter l'algorithme — scikit-image est déjà une dépendance de base
du projet, et sa fonction cite explicitement cette même publication (voir
sa docstring). Pas de raison de dupliquer un code déjà correct et
maintenu par une bibliothèque de référence.

Validation croisée : comparée à ``deepinv.loss.metric.BlurStrength``
(BSD-3-Clause, cite aussi Crété et al. 2007) sur une image test avec flou
gaussien croissant (σ = 0, 1, 2, 4, 8) : forte corrélation monotone, écart
absolu décroissant de ~0.06 (image nette) à ~0.001 (très floue). L'écart
résiduel sur les images nettes s'explique par un choix d'implémentation
différent, pas une erreur : ``skimage.filters.sobel`` calcule une dérivée
directionnelle avec lissage perpendiculaire (Sobel 2-D complet), alors que
``BlurStrength`` utilise un noyau centré brut ``[-1, 0, 1]`` sans lissage
perpendiculaire — plus sensible aux hautes fréquences, donc plus divergent
précisément là où il reste le plus de détails fins à mesurer.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy.typing as npt
from skimage.measure import blur_effect as _blur_effect_skimage


def blur_effect(
    image: npt.NDArray,
    filter_size: int = 11,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Calcule la netteté par la mesure "Blur Effect" (Crété et al., 2007).

    Le papier (et ``skimage.measure.blur_effect``) définissent un score de
    **flou** : ``0`` = image nette, ``1`` = image très floue. Toutes les
    autres métriques de la façade :func:`~eoqual.metrics.nr.sharpness.sharpness`
    (``tenengrad``, ``antonel``, ``blur_kernel``...) suivent la convention
    inverse ("plus élevé = plus net", ``lower_better=False`` dans le
    registre). Pour rester cohérent, cette fonction retourne **1 moins**
    le score brut du papier — le score brut reste accessible via
    ``return_details=True``.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    filter_size : int, optional
        Taille du filtre moyenneur 1-D de reflou (``h_size`` dans le
        papier et dans ``skimage``). Par défaut ``11`` (valeur par défaut
        de ``skimage.measure.blur_effect`` ; le papier utilise ``9``).
        Doit rester identique entre deux images pour être comparable.
    return_details : bool, optional
        Si ``True``, retourne aussi le score brut (échelle du papier) et
        le détail par axe. Par défaut ``False``.

    Returns
    -------
    float
        Score de netteté dans ``[0, 1]`` (``1 - score brut du papier``).
        Valeur plus élevée = image plus nette, cohérent avec le reste de
        la façade ``sharpness``.
    Dict[str, float], optional
        ``{"score", "blur_effect_raw", "b_axis0", "b_axis1"}`` si
        ``return_details=True`` — ``blur_effect_raw`` et ``b_axis0``/
        ``b_axis1`` sont sur l'échelle brute du papier (0=net, 1=flou),
        avant inversion.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = blur_effect(img)
    """
    per_axis = _blur_effect_skimage(image, h_size=filter_size, reduce_func=None)
    raw = float(max(per_axis))
    score = 1.0 - raw
    if not return_details:
        return score
    return score, {
        "score": score,
        "blur_effect_raw": raw,
        "b_axis0": float(per_axis[0]),
        "b_axis1": float(per_axis[1]),
    }
