"""
Métriques d'erreur Full-Reference.

Fonctions exposées : ``mae``, ``mdae``, ``mse``, ``nrmse``, ``rmse``, ``rmse_sw``.
"""
import sys
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check, warn_numpy_only

# Optional dependencies
try:
    from skimage.metrics import normalized_root_mse as _nrmse_skimage
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

try:
    from sewar.full_ref import rmse_sw as _rmse_sw_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False


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
        Seule ``"numpy"`` est implémentée (défaut). Les anciennes
        alternatives (``"skimage"``, ``"metrikz"``, ``"sewar"``) ont été
        retirées après validation : résultat bit-exact et temps de calcul
        équivalent à la version numpy sur image réelle et synthétique (voir
        `REFERENCE_TECHNIQUE.md` §2.1, `formulas.md`). Toute autre valeur
        est acceptée pour compatibilité ascendante : un avertissement est
        émis et le calcul est fait avec ``"numpy"``.
    return_map : bool, optional
        Si ``True``, retourne également la carte d'erreur pixel-à-pixel.
        Par défaut ``False``.

    Returns
    -------
    float
        Valeur de la MSE.
    npt.NDArray, optional
        Carte MSE pixel-à-pixel (uniquement si ``return_map=True``).
    """
    GT, P = initial_check(GT, P)
    if algo != "numpy":
        warn_numpy_only("mse", algo)
    diff = (GT.astype(np.float64) - P.astype(np.float64)) ** 2
    value = float(np.mean(diff))
    return (value, diff) if return_map else value


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
        Seule ``"numpy"`` est implémentée (défaut). ``"metrikz"`` et
        ``"sewar"`` ont été retirés après validation : résultat bit-exact
        et temps de calcul équivalent (voir `REFERENCE_TECHNIQUE.md` §2.1,
        `formulas.md`). ``"image-similarity-measures"`` a également été
        retiré, mais pour une raison différente : ce n'était **pas** la
        même métrique malgré le nom — ISM renvoie la RMSE **normalisée**
        par la dynamique de l'image (``RMSE / (max-min)``, dans ``[0, 1]``)
        et non la RMSE brute en unités de pixel que renvoient numpy/sewar
        (écart observé : ~99,6 % sur une image 8 bits, soit un facteur
        ~256 — cohérent avec cette normalisation). Toute valeur passée ici
        est acceptée pour compatibilité ascendante : un avertissement est
        émis et le calcul est fait avec ``"numpy"``.

    Returns
    -------
    float
        Valeur de la RMSE brute (unités de pixel). Une valeur plus faible
        indique une meilleure qualité.
    """
    GT, P = initial_check(GT, P)
    if algo != "numpy":
        warn_numpy_only("rmse", algo)
    return float(
        np.sqrt(np.mean((GT.astype(np.float64) - P.astype(np.float64)) ** 2))
    )


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
