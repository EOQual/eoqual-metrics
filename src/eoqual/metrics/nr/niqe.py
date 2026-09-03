"""
Naturalness Image Quality Evaluator (NIQE) — No-Reference.
"""
from __future__ import annotations
import numpy.typing as npt


def niqe(P: npt.NDArray, algo: str = "numpy") -> float:
    """
    Calcule le NIQE (Naturalness Image Quality Evaluator).

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
        Score NIQE. Une valeur plus faible indique une meilleure qualité naturelle.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.
    """
    if algo == "numpy":
        from ..backends.niqe.numpy_impl import niqe as _niqe_numpy
        return _niqe_numpy(P)
    raise ValueError(f"'algo' inconnu : {algo!r}")
