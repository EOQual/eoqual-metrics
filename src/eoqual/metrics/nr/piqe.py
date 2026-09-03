"""
Perception based Image Quality Evaluator (PIQE) — No-Reference.
"""
from __future__ import annotations
import numpy.typing as npt


def piqe(P: npt.NDArray, algo: str = "numpy") -> float:
    """
    Calcule le PIQE (Perception based Image Quality Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image d'entrée en niveaux de gris de shape ``(H, W)``.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)

    Returns
    -------
    float
        Score PIQE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.
    """
    if algo == "numpy":
        from ..backends.piqe.numpy_impl import piqe as _piqe
        return _piqe(P)
    raise ValueError(f"'algo' inconnu : {algo!r}")
