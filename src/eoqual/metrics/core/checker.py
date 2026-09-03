"""
Utilitaires de validation et de vérification des images.

Ce module fournit les fonctions de contrôle appliquées systématiquement
avant tout calcul de métrique : compatibilité des shapes, types, versions
de dépendances.
"""

from __future__ import annotations

from typing import Tuple

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
