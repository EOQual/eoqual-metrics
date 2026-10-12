"""
Utilitaires de validation et de vérification des images.

Ce module fournit les fonctions de contrôle appliquées systématiquement
avant tout calcul de métrique : compatibilité des shapes, types, versions
de dépendances.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt


def initial_check(
    GT: npt.NDArray,
    P: npt.NDArray,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Vérifie la compatibilité de deux images avant le calcul d'une métrique.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence (ground truth).
    P : npt.NDArray
        Image traitée / dégradée.

    Returns
    -------
    GT : npt.NDArray
        Image de référence validée.
    P : npt.NDArray
        Image traitée validée.

    Raises
    ------
    ValueError
        Si les deux images n'ont pas le même ``shape``.
    """
    if GT.shape != P.shape:
        raise ValueError(
            f"Les images doivent avoir le même shape. "
            f"GT={GT.shape}, P={P.shape}"
        )
    return GT, P


def versiontuple(version_string: str) -> Tuple[int, ...]:
    """
    Convertit une chaîne de version en tuple comparable.

    Parameters
    ----------
    version_string : str
        Chaîne de version au format ``"X.Y.Z"`` (e.g. ``"1.2.3"``).

    Returns
    -------
    Tuple[int, ...]
        Tuple d'entiers représentant la version (e.g. ``(1, 2, 3)``).

    Examples
    --------
    >>> versiontuple("1.2.3") >= versiontuple("1.2.0")
    True
    """
    return tuple(int(x) for x in version_string.split("."))


def as_channels_last(image: npt.NDArray) -> npt.NDArray:
    """
    Met une image au format ``(H, W, C)`` attendu par
    ``image-similarity-measures``.

    Une image 2D ``(H, W)`` reçoit un axe de canal (``C = 1``) ; une image
    déjà multicanal ``(H, W, C)`` est renvoyée telle quelle. Ajouter l'axe
    à une image ``(H, W, 3)`` produirait ``(H, W, 3, 1)`` : les métriques
    calculées « par canal » le seraient alors sur un mauvais axe (``sam``
    mesurait l'angle entre des scalaires, ``ssim``/``uqi``/``issm``
    échouaient).
    """
    if image.ndim == 2:
        return image[..., None]
    if image.ndim == 3:
        return image
    raise ValueError(f"Image 2D ou 3D (H, W, C) attendue, reçu ndim={image.ndim}.")


def rescale_to_full_scale(
    GT: npt.NDArray,
    P: npt.NDArray,
    full_scale: float,
    data_range: float | None = None,
) -> Tuple[npt.NDArray, npt.NDArray]:
    """
    Ramène deux images sur ``[0, full_scale]`` à partir de la dynamique de
    la référence (origine ``min(GT)``, dynamique ``data_range``, par défaut
    ``max(GT) - min(GT)``).

    Pour les fonctions d'``image-similarity-measures`` dont des constantes
    supposent une dynamique fixe (``issm`` : 12 bits ; ``fsim`` :
    ``T2 = 160`` fixé pour des images 8 bits) : sans cela, leur résultat
    dépend de l'échelle des valeurs (0-255, 0-1, 0-4095...).
    """
    low = float(np.min(GT))
    if data_range is None:
        data_range = float(np.max(GT)) - low
    if data_range <= 0:
        data_range = 1.0
    scale = full_scale / data_range
    return (np.asarray(GT, dtype=np.float64) - low) * scale, (np.asarray(P, dtype=np.float64) - low) * scale
