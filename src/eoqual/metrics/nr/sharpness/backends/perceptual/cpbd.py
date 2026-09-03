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


def _ensure_cpbd_importable() -> None:
    """
    Shim de compatibilité pour importer ``cpbd`` avec un SciPy récent.

    ``cpbd`` (non maintenu depuis 2018) importe historiquement
    ``from scipy.ndimage import imread`` en tête de ``cpbd/compute.py`` ;
    cette fonction a été retirée de SciPy (>= 1.3), ce qui casse l'import
    du module. Plutôt que de patcher le fichier source de ``cpbd`` installé
    sur le disque (l'approche ``postinstall.py`` historique de ce dépôt —
    fragile : à refaire après chaque réinstallation/mise à jour de
    ``cpbd``, invisible pour qui installe la librairie sans lire le
    README), on fournit ici un attribut ``imread`` factice dans
    ``scipy.ndimage`` avant l'import de ``cpbd``. ``cpbd`` ne s'en sert que
    dans son bloc ``if __name__ == "__main__":`` (jamais depuis
    ``cpbd.compute()``, le seul point d'entrée utilisé ici) : ce stub n'est
    donc jamais réellement appelé, il ne fait que satisfaire l'import.

    Sans effet si l'attribut existe déjà (SciPy ancien) ou si la version de
    ``cpbd`` installée a déjà retiré cet import (ex. cpbd >= 1.0.7 récent).
    """
    import scipy.ndimage as _ndimage

    if not hasattr(_ndimage, "imread"):
        def _imread_stub(*_args, **_kwargs):  # pragma: no cover
            raise NotImplementedError(
                "scipy.ndimage.imread n'existe plus (SciPy >= 1.3) ; ce stub "
                "ne sert qu'à satisfaire l'import de cpbd."
            )

        _ndimage.imread = _imread_stub  # type: ignore[attr-defined]


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
        _ensure_cpbd_importable()
        import cpbd as _cpbd  # type: ignore
    except ImportError:
        logger.info("cpbd: librairie indisponible.")
        return None
    return float(_cpbd.compute(image))
