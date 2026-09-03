"""
Métriques de corrélation et statistiques — Full-Reference.

Fonctions exposées : ``ncc``, ``ndp``, ``nmi``, ``scc``, ``uqi``.
"""
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check, versiontuple

try:
    from scipy.spatial import distance as _scipy_distance
    from scipy.spatial.distance import cdist as _cdist
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    from sewar.full_ref import scc as _scc_sewar
    from sewar.full_ref import uqi as _uqi_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False

try:
    from image_similarity_measures.quality_metrics import uiq as _uqi_ism  # type: ignore
    _ISM = True
except ImportError:
    _ISM = False


def ncc(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule le Normalized Cross-Correlation (NCC) — aussi appelé coefficient de Pearson.

    Le résultat est dans ``[-1, 1]``, où ``1`` signifie une corrélation parfaite.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)
        * ``"scipy"``

    Returns
    -------
    float
        Valeur du NCC dans ``[-1, 1]``. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    GT = GT.astype(np.float64)
    P = P.astype(np.float64)

    if algo == "numpy":
        numerator = np.mean((GT - np.mean(GT)) * (P - np.mean(P)))
        denom = np.std(GT) * np.std(P)
        return 0.0 if denom == 0 else float(numerator / denom)
    if algo == "scipy":
        if not _SCIPY:
            raise ImportError("scipy est requis pour algo='scipy'.")
        return float(1.0 - _scipy_distance.correlation(GT.flatten(), P.flatten()))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def ndp(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule le Normalized Dot Product (NDP) — aussi appelé angle cosinusoïdal (VCAD).

    Le résultat est dans ``[-1, 1]`` où ``1`` = vecteurs colinéaires positifs,
    ``0`` = orthogonaux, ``-1`` = opposés.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)
        * ``"scipy_1"``
        * ``"scipy_2"``
        * ``"sklearn"``

    Returns
    -------
    float
        Valeur du NDP dans ``[-1, 1]``. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    GT = GT.astype(np.float64)
    P = P.astype(np.float64)

    if algo == "numpy":
        return float(
            np.dot(GT.flatten(), P.flatten())
            / (np.linalg.norm(GT) * np.linalg.norm(P))
        )
    if algo == "scipy_1":
        if not _SCIPY:
            raise ImportError("scipy est requis pour algo='scipy_1'.")
        return float(1.0 - _scipy_distance.cosine(GT.flatten(), P.flatten()))
    if algo == "scipy_2":
        if not _SCIPY:
            raise ImportError("scipy est requis pour algo='scipy_2'.")
        return float(
            1.0 - _cdist(GT.reshape(1, -1), P.reshape(1, -1), "cosine")[0, 0]
        )
    if algo == "sklearn":
        if not _SKLEARN:
            raise ImportError("scikit-learn est requis pour algo='sklearn'.")
        return float(_cosine_similarity(GT.reshape(1, -1), P.reshape(1, -1))[0, 0])
    raise ValueError(f"'algo' inconnu : {algo!r}")


def nmi(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "skimage",
) -> float:
    """
    Calcule l'Information Mutuelle Normalisée (NMI).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"skimage"`` (défaut, nécessite scikit-image >= 0.19)

    Returns
    -------
    float
        Valeur de la NMI. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si scikit-image < 0.19 est utilisé.
    """
    GT, P = initial_check(GT, P)
    if algo == "skimage":
        import skimage  # type: ignore
        if versiontuple(skimage.__version__) < versiontuple("0.19"):
            raise ImportError("scikit-image >= 0.19 est requis pour nmi.")
        from skimage.metrics import normalized_mutual_information as _nmi_ski
        return float(_nmi_ski(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def scc(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
) -> float:
    """
    Calcule le Spatial Correlation Coefficient (SCC).

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
        Valeur du SCC. Une valeur plus élevée indique une meilleure qualité.

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
        return float(_scc_sewar(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def uqi(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "metrikz",
) -> Optional[float]:
    """
    Calcule l'Universal Image Quality Index (UQI).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"metrikz"`` (défaut)
        * ``"sewar"``
        * ``"image-similarity-measures"``

    Returns
    -------
    float or None
        Valeur de l'UQI, ou ``None`` si la librairie requise est indisponible.
        Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
#    _ALGO_GRAY_ONLY = {
#        "metrikz", "sewar", "image-similarity-measures"
#    }
#    _ALGO_RGB_CAPABLE = {
#        "metrikz", "sewar"
#    }
#
#    if GT.ndim > 3:
#        logger.error(f"L'image doit être 2D ou 3D, reçu shape={GT.shape}.")
#        raise ValueError(f"L'image doit être 2D ou 3D, reçu shape={GT.shape}.")
#    if GT.ndim == 3 and algo in _ALGO_GRAY_ONLY - _ALGO_RGB_CAPABLE:
#        logger.warning(f"uqi/{algo} — nécessite une image monobande.")
#        return None
#    if GT.ndim == 2 and algo in _ALGO_RGB_CAPABLE - _ALGO_GRAY_ONLY:
#        logger.warning(f"uqi/{algo} — nécessite une image RGB.")
#        return None

    GT, P = initial_check(GT, P)
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import uqi as _uqi_metrikz
        return float(_uqi_metrikz(GT, P))
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_uqi_sewar(GT, P))
    if algo == "image-similarity-measures":
        if not _ISM:
            return None
        return float(_uqi_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    raise ValueError(f"'algo' inconnu : {algo!r}")
