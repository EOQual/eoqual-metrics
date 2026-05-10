"""
Perception based Image Quality Evaluator (PIQE) — No-Reference.
"""
from __future__ import annotations
import numpy.typing as npt


def piqe(P: npt.NDArray, algo: str = "eadcat_impl") -> float:
    """
    Calcule le PIQE (Perception based Image Quality Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image d'entrée en niveaux de gris de shape ``(H, W)``.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"eadcat_impl"`` (défaut)

    Returns
    -------
    float
        Score PIQE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.
    """
    if algo == "eadcat_impl":
        from ..backends.piqe.eadcat_impl import piqe as _piqe
        return float(_piqe(P))
    raise ValueError(f"'algo' inconnu : {algo!r}")
