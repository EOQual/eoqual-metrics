"""
RECO — Relative Polar Edge Coherence.

Implémentation propre, écrite à partir de la description publiée de la
méthode (voir Référence) — remplace l'implémentation précédemment vendored
(`backends/reco/polar_impl.py`, GPL-3.0, voir `THIRD_PARTY_LICENSES.md`).

Principe
--------
La cohérence de phase entre deux harmoniques circulaires de Laguerre-Gauss
d'ordre 1 et 3 (``L1,0`` et ``L3,0``) mesure la netteté locale d'un bord :
en un point de bord idéal, la phase de la réponse d'ordre 3 est en phase
avec 3× la phase de la réponse d'ordre 1 (les deux répondent au même bord
sous-jacent), ce qui donne une magnitude de cohérence proche de 1 ; le
bruit ou une texture sans structure de bord donnent une cohérence proche
de 0.

1. Convoler l'image avec les deux filtres complexes ``L1,0``/``L3,0``.
2. Cohérence de bord en un point : produit des magnitudes pondéré par le
   cosinus de l'écart de phase (``angle(y3) - 3*angle(y1)``).
3. Score de cohérence globale (ECO) = somme de cette cohérence sur
   l'image entière.
4. RECO = ratio ``ECO(image traitée) / ECO(image de référence)`` (métrique
   à référence réduite — un seul nombre suffit côté référence).

Référence
---------
Baroncini, V., Capodiferro, L., Di Claudio, E. D., & Jacovitti, G. (2009).
*The polar edge coherence: a quasi blind metric for video quality
assessment*. EUSIPCO 2009, Glasgow, 564-568.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve


def _laguerre_gauss_harmonic(order: int, size: int, sigma: float) -> npt.NDArray:
    """
    Génère le noyau harmonique circulaire de Laguerre-Gauss d'ordre
    ``order`` (0 dans la coordonnée radiale), de taille ``size × size``.

    ``L_{n,0}(r, γ) = -c_n / (σ√π) · exp(-r²/2σ²) · (r/σ)^n · exp(-i·n·γ)``
    avec ``c_1 = 1`` et ``c_3 = 1/6`` (facteurs de normalisation usuels
    pour ces deux ordres).
    """
    coords = np.linspace(-size / 2.0, size / 2.0, size)
    xx, yy = np.meshgrid(coords, coords)
    r = np.sqrt(xx ** 2 + yy ** 2)
    gamma = np.arctan2(yy, xx)
    c_n = 1.0 if order == 1 else (1.0 / 6.0 if order == 3 else 1.0)
    radial = (r / sigma) ** order
    envelope = np.exp(-r ** 2 / (2 * sigma ** 2))
    return -c_n / (sigma * np.sqrt(np.pi)) * envelope * radial * np.exp(-1j * order * gamma)


def _edge_coherence_map(image: npt.NDArray, size: int = 17, sigma: float = 2.0) -> npt.NDArray:
    """
    Carte de cohérence de bord polaire, même shape que ``image``.
    """
    l1 = _laguerre_gauss_harmonic(1, size, sigma)
    l3 = _laguerre_gauss_harmonic(3, size, sigma)

    def _complex_convolve(img: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
        return convolve(img, np.real(kernel)) + 1j * convolve(img, np.imag(kernel))

    y1 = _complex_convolve(image, l1)
    y3 = _complex_convolve(image, l3)
    return -(np.abs(y3) / np.abs(y1)) * np.cos(np.angle(y3) - 3 * np.angle(y1))


def _edge_coherence_score(image: npt.NDArray, size: int = 17, sigma: float = 2.0) -> float:
    """Somme de la carte de cohérence de bord — un seul score par image."""
    l1 = _laguerre_gauss_harmonic(1, size, sigma)
    l3 = _laguerre_gauss_harmonic(3, size, sigma)

    def _complex_convolve(img: npt.NDArray, kernel: npt.NDArray) -> npt.NDArray:
        return convolve(img, np.real(kernel)) + 1j * convolve(img, np.imag(kernel))

    y1 = _complex_convolve(image, l1)
    y3 = _complex_convolve(image, l3)
    return float(np.sum(-(np.abs(y3) * np.abs(y1)) * np.cos(np.angle(y3) - 3 * np.angle(y1))))


def reco(GT: npt.NDArray, P: npt.NDArray, regularization: float = 1.0) -> float:
    """
    Calcule le Relative Polar Edge Coherence (RECO) entre deux images.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée, même shape.
    regularization : float, optional
        Constante additive évitant la division par une cohérence de
        référence nulle/négative (voir papier original, non spécifiée
        précisément — ``1.0`` par convention). Par défaut ``1.0``.

    Returns
    -------
    float
        Ratio ``ECO(P) / ECO(GT)``. Une valeur plus élevée (proche ou
        au-dessus de 1) indique une cohérence de bord de ``P`` proche de
        celle de ``GT``.

    Examples
    --------
    >>> import numpy as np
    >>> ref = np.random.rand(128, 128)
    >>> score = reco(ref, ref)
    """
    ref = GT.astype(np.float64)
    dist = P.astype(np.float64)
    eco_ref = _edge_coherence_score(ref)
    eco_dist = _edge_coherence_score(dist)
    return (eco_dist + regularization) / (eco_ref + regularization)
