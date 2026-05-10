"""
Métriques d'erreur Full-Reference.

Fonctions exposées : ``mae``, ``mdae``, ``mse``, ``nrmse``, ``rmse``, ``rmse_sw``.
"""
import sys
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check

# Optional dependencies
try:
    from skimage.metrics import mean_squared_error as _mse_skimage
    from skimage.metrics import normalized_root_mse as _nrmse_skimage
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

try:
    from sewar.full_ref import mse as _mse_sewar
    from sewar.full_ref import rmse as _rmse_sewar
    from sewar.full_ref import rmse_sw as _rmse_sw_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False

try:
    from image_similarity_measures.quality_metrics import rmse as _rmse_ism  # type: ignore
    _ISM = True
except ImportError:
    _ISM = False


def mae(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule la Mean Absolute Error (MAE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)

    Returns
    -------
    float
        Valeur de la MAE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "numpy":
        return float(np.mean(np.abs(GT.astype(np.float64) - P.astype(np.float64))))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def mdae(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule la Median Absolute Error (MdAE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)

    Returns
    -------
    float
        Valeur de la MdAE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "numpy":
        return float(np.median(np.abs(GT.astype(np.float64) - P.astype(np.float64))))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def mse(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
    return_map: bool = False,
) -> Union[float, Tuple[float, npt.NDArray]]:
    """
    Calcule la Mean Squared Error (MSE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)
        * ``"skimage"``
        * ``"metrikz"``
        * ``"sewar"``
    return_map : bool, optional
        Si ``True`` et ``algo="numpy"``, retourne également la carte d'erreur
        pixel-à-pixel. Par défaut ``False``.

    Returns
    -------
    float
        Valeur de la MSE.
    npt.NDArray, optional
        Carte MSE pixel-à-pixel (uniquement si ``return_map=True`` et ``algo="numpy"``).

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    diff = (GT.astype(np.float64) - P.astype(np.float64)) ** 2

    if algo == "numpy":
        value = float(np.mean(diff))
        return (value, diff) if return_map else value
    if algo == "skimage":
        if not _SKIMAGE:
            raise ImportError("scikit-image est requis pour algo='skimage'.")
        return float(_mse_skimage(GT, P))
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import mse as _mse_metrikz
        return float(_mse_metrikz(GT, P))
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_mse_sewar(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def nrmse(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "skimage",
) -> float:
    """
    Calcule la Normalized Root Mean-Squared Error (NRMSE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"skimage"`` (défaut)

    Returns
    -------
    float
        Valeur de la NRMSE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si scikit-image n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if algo == "skimage":
        if not _SKIMAGE:
            raise ImportError("scikit-image est requis pour algo='skimage'.")
        return float(_nrmse_skimage(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def rmse(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule la Root Mean Squared Error (RMSE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)
        * ``"metrikz"``
        * ``"sewar"``
        * ``"image-similarity-measures"``

    Returns
    -------
    float
        Valeur de la RMSE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "numpy":
        return float(
            np.sqrt(np.mean((GT.astype(np.float64) - P.astype(np.float64)) ** 2))
        )
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import rmse as _rmse_metrikz
        return float(_rmse_metrikz(GT, P))
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_rmse_sewar(GT, P))
    if algo == "image-similarity-measures":
        if not _ISM:
            raise ImportError("image_similarity_measures est requis.")
        return float(_rmse_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def rmse_sw(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
) -> float:
    """
    Calcule la Root Mean Squared Error avec fenêtre glissante (RMSE-SW).

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
        Valeur de la RMSE-SW. Une valeur plus faible indique une meilleure qualité.

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
        return float(_rmse_sw_sewar(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")
