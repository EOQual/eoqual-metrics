"""
Mesure de netteté CPBD (Cumulative Probability of Blur Detection).

Référence
---------
Narvekar, N. D., & Karam, L. J. (2011). *A no-reference image blur metric based on the
cumulative probability of blur detection*. IEEE Transactions on Image Processing.
"""
from __future__ import annotations
from typing import Optional
import numpy.typing as npt
from loguru import logger


def sharpness_cpbd(image: npt.NDArray) -> Optional[float]:
    """
    Calcule la netteté CPBD (Cumulative Probability of Blur Detection).

    Nécessite la librairie ``cpbd``.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.

    Returns
    -------
    float or None
        Score CPBD dans ``[0, 1]``. Valeur plus élevée = image plus nette.
        ``None`` si la librairie ``cpbd`` est indisponible.
    """
    try:
        import cpbd as _cpbd  # type: ignore
    except ImportError:
        logger.info("cpbd: librairie indisponible.")
        return None
    return float(_cpbd.compute(image))
