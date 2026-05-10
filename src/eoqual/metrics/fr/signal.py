"""
Métriques basées sur le rapport signal/bruit — Full-Reference.

Fonctions exposées : ``snr``, ``psnr``, ``psnrb``, ``psnrc``, ``wsnr``.
"""
from typing import Optional

import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check
from .error import mse

try:
    from skimage.metrics import peak_signal_noise_ratio as _psnr_skimage
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

try:
    from sewar.full_ref import psnr as _psnr_sewar
    from sewar.full_ref import psnrb as _psnrb_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False

try:
    from image_similarity_measures.quality_metrics import psnr as _psnr_ism  # type: ignore
    _ISM = True
except ImportError:
    _ISM = False


def snr(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule le Signal-to-Noise Ratio (SNR).

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

    Returns
    -------
    float
        Valeur du SNR en dB. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "numpy":
        signal_power = (GT.astype(np.float64) ** 2).mean()
        with np.errstate(divide="ignore"):
            return float(10.0 * np.log10(signal_power / mse(GT, P, algo="numpy")))
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import snr as _snr_metrikz
        return float(_snr_metrikz(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def psnr(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule le Peak Signal-to-Noise Ratio (PSNR).

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
        * ``"image-similarity-measures"``

    Returns
    -------
    float
        Valeur du PSNR en dB. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "numpy":
        rng = float(np.abs(GT.max() - GT.min()))
        with np.errstate(divide="ignore"):
            return float(10.0 * np.log10(rng ** 2 / mse(GT, P, algo="numpy")))
    if algo == "skimage":
        if not _SKIMAGE:
            raise ImportError("scikit-image est requis pour algo='skimage'.")
        pixel_diff = float(np.max(np.maximum(GT, P)) - np.min(np.minimum(GT, P)))
        return float(_psnr_skimage(GT, P, data_range=pixel_diff))
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import psnr as _psnr_metrikz
        return float(_psnr_metrikz(GT, P))
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_psnr_sewar(GT, P))
    if algo == "image-similarity-measures":
        if not _ISM:
            raise ImportError("image_similarity_measures est requis.")
        return float(_psnr_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def psnrb(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
) -> float:
    """
    Calcule le PSNR avec facteur d'effet de blocs (PSNR-B).

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
        Valeur du PSNR-B en dB. Une valeur plus élevée indique une meilleure qualité.

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
        return float(_psnrb_sewar(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def psnrc(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> Optional[float]:
    """
    Calcule le PSNR couleur (PSNR-C) sur les 3 canaux RGB séparément.

    Le score final est la moyenne des PSNR par canal.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence de shape ``(H, W, 3)``.
    P : npt.NDArray
        Image traitée / dégradée de shape ``(H, W, 3)``.
    algo : str, optional
        Algorithme à utiliser (mêmes options que :func:`psnr`).
        Par défaut ``"numpy"``.

    Returns
    -------
    float
        Valeur du PSNR-C en dB. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si les images n'ont pas 3 canaux ou si ``algo`` est inconnu.
    """
    if GT.ndim != 3 or GT.shape[2] != 3:
        logger.warning("psnrc nécessite des images RGB de shape (H, W, 3).")
        return None
    return float(np.mean([psnr(GT[..., c], P[..., c], algo=algo) for c in range(3)]))


def wsnr(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "metrikz",
) -> float:
    """
    Calcule le Weighted Signal-to-Noise Ratio (WSNR).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"metrikz"`` (défaut)

    Returns
    -------
    float
        Valeur du WSNR. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import wsnr as _wsnr_metrikz
        return float(_wsnr_metrikz(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")

#EOF