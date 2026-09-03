"""
Blind/Referenceless Image Spatial Quality Evaluator (BRISQUE) — No-Reference.
"""
from __future__ import annotations

import numpy.typing as npt
from skimage.color import gray2rgb

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


def brisque(P: npt.NDArray, algo: str = "numpy") -> float:
    """
    Calcule le BRISQUE (Blind/Referenceless Image Spatial Quality Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image d'entrée en niveaux de gris de shape ``(H, W)``.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut) — sans dépendance externe (pas de libsvm)
        * ``"image-quality"`` — librairie imquality (extra ``svm``)
        * ``"brisque"`` — librairie brisque, RGB (extra ``svm``)

    Returns
    -------
    float
        Score BRISQUE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.
    ImportError
        Si la dépendance requise par l'algo choisi n'est pas installée.
    """
    if algo == "numpy":
        from ..backends.brisque.numpy_impl import brisque as _brisque_numpy
        return _brisque_numpy(P)

    if algo == "image-quality":
        if not _IMQUALITY:
            raise ImportError(
                "imquality est requis pour algo='image-quality' : "
                "pip install \"eoqual-metrics[svm]\" (voir README.md)."
            )
        return float(_brisque_imquality.score(P))

    if algo == "brisque":
        if not _BRISQUE_LIB_OK:
            raise ImportError(
                "la librairie brisque est requise pour algo='brisque' : "
                "pip install \"eoqual-metrics[svm]\" (voir README.md)."
            )
        obj = _BRISQUE_LIB(url=False)
        return float(obj.score(gray2rgb(P)))

    raise ValueError(f"'algo' inconnu : {algo!r}")
