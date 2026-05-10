"""
Mesure de netteté via pydom.
"""
from __future__ import annotations
from typing import Optional
import numpy.typing as npt
from loguru import logger


def sharpness_pydom(image: npt.NDArray) -> Optional[float]:
    """
    Calcule la netteté via la librairie pydom.

    Parameters
    ----------
    image : npt.NDArray
        Image d'entrée de shape ``(H, W)`` ou ``(H, W, 3)``.

    Returns
    -------
    float or None
        Score de netteté. Valeur plus élevée = image plus nette.
        ``None`` si la librairie ``pydom`` est indisponible.
    """
    try:
        from dom import DOM  # type: ignore
    except ImportError:
        logger.warning("pydom: DOM.sharpness_measure  indisponible.")
        return None
#FIXME: width=2, sharpness_threshold=2.0 sont des hyperparamètres à exposer dans la config
    return float(DOM().sharpness_measure(image, width=2, sharpness_threshold=2.0))

#EOF