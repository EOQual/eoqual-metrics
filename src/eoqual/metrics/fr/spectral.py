"""
Métriques spectrales et de télédétection — Full-Reference.

Fonctions exposées : ``ergas``, ``rase``, ``sam``, ``sre``.
"""
import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check

try:
    from sewar.full_ref import ergas as _ergas_sewar
    from sewar.full_ref import rase as _rase_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False

try:
    from image_similarity_measures.quality_metrics import (  # type: ignore
        sam as _sam_ism,
        sre as _sre_ism,
    )
    _ISM = True
except ImportError:
    _ISM = False


def ergas(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
) -> float:
    """
    Calcule l'Erreur Relative Globale Adimensionnelle de Synthèse (ERGAS).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"sewar"`` (défaut)

    Returns
    -------
    float
        Valeur de l'ERGAS. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si sewar n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_ergas_sewar(GT, P))
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def rase(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
) -> float:
    """
    Calcule la Relative Average Spectral Error (RASE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"sewar"`` (défaut)

    Returns
    -------
    float
        Valeur de la RASE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si sewar n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_rase_sewar(GT, P))
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def sam(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "image-similarity-measures",
) -> float:
    """
    Calcule le Spectral Angle Mapper (SAM).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"image-similarity-measures"`` (défaut)

    Returns
    -------
    float
        Valeur du SAM. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si ``image-similarity-measures`` n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if algo == "image-similarity-measures":
        if not _ISM:
            raise ImportError(
                "image-similarity-measures est requis pour cet algo : "
                "pip install image-similarity-measures --no-deps (voir README.md)."
            )
        return float(_sam_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def sre(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "image-similarity-measures",
) -> float:
    """
    Calcule le Signal to Reconstruction Error Ratio (SRE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"image-similarity-measures"`` (défaut)

    Returns
    -------
    float
        Valeur du SRE. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si ``image-similarity-measures`` n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if algo == "image-similarity-measures":
        if not _ISM:
            raise ImportError(
                "image-similarity-measures est requis pour cet algo : "
                "pip install image-similarity-measures --no-deps (voir README.md)."
            )
        return float(_sre_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")

#EOF