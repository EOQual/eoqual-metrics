"""
Métriques spectrales et de télédétection — Full-Reference.

Fonctions exposées : ``ergas``, ``rase``, ``sam``, ``sre``.
"""
import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import as_channels_last, initial_check

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
        Angle spectral moyen, en degrés. Plus il est petit, plus les
        spectres sont proches (0 = spectres colinéaires).

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu, si les images sont incompatibles, ou
        si elles n'ont qu'un canal (l'angle entre deux scalaires positifs
        est toujours nul : SAM n'a de sens qu'en multicanal).
    ImportError
        Si ``image-similarity-measures`` n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if GT.ndim != 3 or GT.shape[2] < 2:
        raise ValueError(
            f"SAM nécessite des images multicanal (H, W, C >= 2), reçu shape={GT.shape}."
        )
    if algo == "image-similarity-measures":
        if not _ISM:
            raise ImportError(
                "image-similarity-measures est requis pour cet algo : "
                "pip install image-similarity-measures --no-deps (voir README.md)."
            )
        # float64 : l'arc cosinus est très mal conditionné près de 1
        # (spectres presque colinéaires) ; en float32, deux images
        # identiques donnaient ~0.006° au lieu de 0.
        return float(_sam_ism(GT.astype(np.float64), P.astype(np.float64)))
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
        return float(_sre_ism(as_channels_last(GT), as_channels_last(P)))
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")

#EOF