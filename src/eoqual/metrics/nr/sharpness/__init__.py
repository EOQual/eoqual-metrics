"""
Module de métriques de netteté — No-Reference.

Ce module regroupe toutes les métriques d'estimation de la netteté d'une image,
organisées par famille algorithmique :

* **Gradient** : ``tenengrad``, ``laplacian``, ``sobel``
* **Phase** : ``lpc_si``, ``sharpness_index``, ``psi``
* **Spectral** : ``s3``, ``mtf``
* **Perceptual** (librairie externe) : ``cpbd``
* **Satellite** (calibré, remote sensing) : ``brenner``, ``fft``, ``antonel``,
  ``blur_kernel``, ``wavelet``, ``aem``, ``sasbem`` — voir
  :mod:`eoqual.metrics.nr.sharpness.backends.satellite`.
  ``aem``/``sasbem`` sont, avec ``mtf``,
  les seules métriques physiquement calibrées (MTF, ISO 12233) du module ;
  les autres sont des indicateurs rapides, non calibrés, sensibles au
  contenu de la scène.

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

        **Perceptual (librairie externe) :**

        * ``"cpbd"`` — Cumulative Probability of Blur Detection

        **Satellite (calibré, remote sensing) :**

        * ``"brenner"`` — variations abruptes entre pixels espacés
        * ``"fft"`` — ratio d'énergie haute fréquence
        * ``"antonel"`` — décroissance de gradient robuste à l'exposition
        * ``"blur_kernel"`` — norme du noyau de flou estimé (PSF)
        * ``"wavelet"`` — énergie des sous-bandes haute fréquence
        * ``"aem"`` — MTF@Nyquist par arêtes naturelles (slanted-edge, ISO 12233)
        * ``"sasbem"`` — AEM + filtrage statistique robuste (MAD) et indice de fiabilité

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

    # --- Perceptual (external lib) ---
    if algo == "cpbd":
        from .backends.perceptual.cpbd import sharpness_cpbd
        return sharpness_cpbd(P)

    # --- Satellite (calibré, remote sensing) ---
    if algo == "brenner":
        from .backends.satellite.brenner import brenner
        return brenner(P)
    if algo == "fft":
        from .backends.satellite.fft import fft_sharpness
        return fft_sharpness(P)
    if algo == "antonel":
        from .backends.satellite.antonel import antonel
        return antonel(P)
    if algo == "blur_kernel":
        from .backends.satellite.blur_kernel import blur_kernel
        return blur_kernel(P)
    if algo == "wavelet":
        from .backends.satellite.wavelet import wavelet
        return wavelet(P)
    if algo == "aem":
        from .backends.satellite.aem import aem
        return aem(P)
    if algo == "sasbem":
        from .backends.satellite.sasbem import sasbem
        return sasbem(P)

    raise ValueError(
        f"'algo' inconnu : {algo!r}. "
        f"Valeurs valides : tenengrad, laplacian, sobel, lpc_si, sharpness_index, "
        f"psi, s3, mtf, cpbd, "
        f"brenner, fft, antonel, blur_kernel, wavelet, aem, sasbem."
    )


__all__ = ["sharpness"]
