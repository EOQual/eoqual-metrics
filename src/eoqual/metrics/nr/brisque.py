"""
Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) — No-Reference.
"""
from __future__ import annotations
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

import numpy.typing as npt
from loguru import logger
from skimage.color import gray2rgb

from ..core.checker import versiontuple

def _libsvm_available() -> bool:
    if sys.platform == "win32":
        libsvm = "libsvm.dll"
    elif sys.platform == "darwin":
        libsvm = "libsvm.dylib"
    elif sys.platform == "linux":
        libsvm = "libsvm.so.2"
    else:
        warnings.warn(f"Plateforme [{sys.platform}] non gérée pour libsvm.")
        return False
    root = Path(os.path.dirname(os.path.realpath(__file__)))
    return any(f.name == libsvm for f in root.rglob("*"))

_LIBSVM = _libsvm_available()

try:
    from imquality import brisque as _brisque_imquality  # type: ignore
    _IMQUALITY = True
except ImportError:
    _IMQUALITY = False

try:
    from brisque import BRISQUE as _BRISQUE_LIB  # type: ignore
    _BRISQUE_LIB_OK = True
except ImportError:
    _BRISQUE_LIB_OK = False

if _LIBSVM:
    try:
        from ..backends.brisque.libsvm_impl.brisque import brisque as _brisque_libsvm
    except Exception:
        _LIBSVM = False

try:
    from ..backends.brisque.sklearn_impl import brisqueScore as _brisque_sklearn
    _SKLEARN_BRISQUE = True
except Exception:
    _SKLEARN_BRISQUE = False


def brisque(P: npt.NDArray, algo: str = "libsvm_impl") -> Optional[float]:
    """
    Calcule le BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image d'entrée en niveaux de gris de shape ``(H, W)``.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"libsvm_impl"`` (défaut) — nécessite libsvm natif
        * ``"sklearn_impl"`` — nécessite scikit-learn == 0.22
        * ``"image-quality"`` — librairie imquality
        * ``"brisque"`` — librairie brisque (RGB)

    Returns
    -------
    float or None
        Score BRISQUE. Une valeur plus faible indique une meilleure qualité.
        ``None`` si la dépendance requise est indisponible.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.
    """
    if algo == "image-quality":
        if not _IMQUALITY:
            logger.warning("image-quality: librairie imquality indisponible.")
            return None
        return float(_brisque_imquality.score(P))

    if algo == "libsvm_impl":
        if not _LIBSVM:
            logger.warning("brisque/libsvm_impl: libsvm est manquant.")
            return None
        return float(_brisque_libsvm(P))

    if algo == "sklearn_impl":
        if not _SKLEARN_BRISQUE:
            logger.warning("brisque/sklearn_impl: indisponible.")
            return None
        import sklearn
        if versiontuple(sklearn.__version__) != versiontuple("0.22"):
            logger.warning(f"sklearn_impl: version [{sklearn.__version__}] <> 0.22.")
            return None
        return float(_brisque_sklearn(P))

    if algo == "brisque":
        if not _BRISQUE_LIB_OK:
            logger.warning("brisque: librairie brisque indisponible.")
            return None
        obj = _BRISQUE_LIB(url=False)
        try:
            return float(obj.score(gray2rgb(P)))
        except AttributeError:
            logger.warning("brisque: erreur libsvm (scipy.ndarray introuvable).")
            return None

    raise ValueError(f"'algo' inconnu : {algo!r}")
