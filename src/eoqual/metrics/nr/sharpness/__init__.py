"""
Module de métriques de netteté — No-Reference.

Ce module regroupe toutes les métriques d'estimation de la netteté d'une image,
organisées par famille algorithmique :

* **Gradient** : ``tenengrad``, ``laplacian``, ``sobel``
* **Phase** : ``lpc_si``, ``sharpness_index``, ``psi``
* **Spectral** : ``s3``, ``mtf``
* **Perceptual** (librairies externes) : ``cpbd``, ``sharpness_1``, ``pydom``

La fonction façade :func:`sharpness` offre un point d'entrée unifié.
"""
from typing import Optional, Union

import numpy.typing as npt
from loguru import logger


def sharpness(
    P: npt.NDArray,
    algo: str = "tenengrad",
) -> Optional[float]:
    """
    Calcule la netteté d'une image (No-Reference).

    Façade unifiée pour toutes les métriques de netteté disponibles.
    Toutes les fonctions sous-jacentes acceptent une image en niveaux de gris
    (``float32`` ou ``float64``, valeurs dans ``[0, 1]`` recommandées).

    Parameters
    ----------
    P : npt.NDArray
        Image d'entrée en niveaux de gris de shape ``(H, W)``.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        **Gradient-based :**

        * ``"tenengrad"`` (défaut) — énergie du gradient Sobel (Krotkov 1987)
        * ``"laplacian"`` — variance du Laplacien
        * ``"sobel"`` — moyenne absolue du gradient Sobel

        **Phase-based :**

        * ``"lpc_si"`` — Local Phase Coherence Sharpness Index (Kovesi / Wang)
        * ``"sharpness_index"`` — Sharpness Index basé sur la variation totale (Moisan/Blanchet)
        * ``"psi"`` — Perceptual Sharpness Index (Feichtenhofer 2013)

        **Spectral :**

        * ``"s3"`` — Spectral & Spatial Sharpness (S3)
        * ``"mtf"`` — MTF estimée par profil de bord

        **Perceptual (librairies externes) :**

        * ``"cpbd"`` — Cumulative Probability of Blur Detection
        * ``"sharpness_1"`` — implémentation personnalisée
        * ``"pydom"`` — pydom

    Returns
    -------
    float or None
        Score de netteté. Une valeur plus élevée indique une image plus nette
        (``lower_better=False``).
        ``None`` si la dépendance requise est indisponible.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = sharpness(img, algo='tenengrad')
    """
    # --- Gradient-based ---
    if algo == "tenengrad":
        from .backends.gradient.tenengrad import tenengrad
        return tenengrad(P)
    if algo == "laplacian":
        from .backends.gradient.laplacian import laplacian
        return laplacian(P)
    if algo == "sobel":
        from .backends.gradient.sobel import sobel_sharpness
        return sobel_sharpness(P)

    # --- Phase-based ---
    if algo == "lpc_si":
        from .backends.phase.lpc_si import lpc_si
        return lpc_si(P)
    if algo == "sharpness_index":
        from .backends.phase.sharpness_index import sharpness_index
        return sharpness_index(P)
    if algo == "psi":
        from .backends.phase.psi import psi
        return psi(P)

    # --- Spectral ---
    if algo == "s3":
        from .backends.spectral.s3 import s3
        return s3(P)
    if algo == "mtf":
        from .backends.spectral.mtf import mtf
        return mtf(P)

    # --- Perceptual (external libs) ---
    if algo == "cpbd":
        from .backends.perceptual.cpbd import sharpness_cpbd
        return sharpness_cpbd(P)
    if algo == "sharpness_1":
        from .backends.perceptual.sharpness_1 import sharpness as _sharpness_1
        return _sharpness_1(P)
    if algo == "pydom":
        from .backends.perceptual.pydom import sharpness_pydom
        return sharpness_pydom(P)

    raise ValueError(
        f"'algo' inconnu : {algo!r}. "
        f"Valeurs valides : tenengrad, laplacian, sobel, lpc_si, sharpness_index, "
        f"psi, s3, mtf, cpbd, sharpness_1, pydom."
    )


__all__ = ["sharpness"]
