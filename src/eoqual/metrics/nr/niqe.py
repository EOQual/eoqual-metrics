"""
Naturalness Image Quality Evaluator (NIQE) — No-Reference.
"""
from __future__ import annotations
import numpy.typing as npt


def niqe(P: npt.NDArray, algo: str = "eadcat_impl") -> float:
    """
    Calcule le NIQE (Naturalness Image Quality Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image d'entrée en niveaux de gris de shape ``(H, W)``.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"eadcat_impl"`` (défaut) — portage du code MATLAB EadCat
        * ``"programtalk_impl"`` — portage alternatif

    Returns
    -------
    float
        Score NIQE. Une valeur plus faible indique une meilleure qualité naturelle.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.
    """
    if algo == "eadcat_impl":
        from ..backends.niqe.eadcat_impl import niqe as _niqe_eadcat
        return float(_niqe_eadcat(P))
    if algo == "programtalk_impl":
        from ..backends.niqe.programtalk_impl import niqe as _niqe_pt
        return float(_niqe_pt(P))
    raise ValueError(f"'algo' inconnu : {algo!r}")
