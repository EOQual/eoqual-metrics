#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Indice de netteté SI (Sharpness Index), fondé sur la cohérence de phase
globale de la transformée de Fourier.

On compare la variation totale TV(u) de l'image à celle d'un champ
gaussien de même spectre d'amplitude mais de phase aléatoire : plus TV(u)
est petite devant cette référence, plus la phase est cohérente (contours
nets). SI décroît avec le flou, le bruit et le ringing ; c'est un score
ordinal sans référence, à comparer entre versions d'une même scène, à
taille égale.

G. Blanchet, L. Moisan, B. Rougé, "Measuring the global phase coherence of an image" (ICIP 2008).

G. Blanchet, L. Moisan, "An explicit Sharpness Index related to Global Phase Coherence" (ICASSP 2012).

A. Leclaire, L. Moisan, "No-reference image quality assessment and blind deblurring with sharpness metrics exploiting Fourier phase information" (Journal of Mathematical Imaging and Vision, 2015).

L. Moisan, "Periodic plus smooth image decomposition" (Journal of Mathematical Imaging and Vision, 2011).

Implémentation de l'auteur, écrite à partir des formules de ces
articles ; décomposition périodique et
déquantification fournies par ``eoqual.core.fourier`` ; les moments μ et σ sont validés
contre une simulation Monte-Carlo (tests). Implémentation MATLAB de
référence de l'auteur de la méthode :
https://www.math.u-psud.fr/~moisan/soft/sharpness_index/
"""
__author__     = "Olivier Amram"
__copyright__  = "Copyright (c) 2026 Olivier Amram"
__credits__    = "Olivier Amram"
__license__    = "MIT"
__version__    = "20261011"
__maintainer__ = "Olivier Amram"
__email__      = "olivier.amram@free.fr"
__status__     = "Production" # Prototype / Development / Production
__all__        = [
    'sharpness_index',
    'sharpness_stats',
] # Liste des objets exportés par le module

# IMPORTS TIERS
import numpy as np
from scipy.special import log_ndtr

from eoqual.core.fourier import dequantize, perdecomp


def _check_2d(image: np.ndarray) -> np.ndarray:
    u = np.asarray(image, dtype=np.float64)
    if u.ndim != 2:
        raise ValueError(f"image 2D attendue, reçu ndim={u.ndim} (traiter les canaux séparément)")
    return u


# Décomposition périodique + lisse (Moisan, 2011) et déquantification
# (translation d'un demi-pixel) : briques partagées d'eoqual-core.
_perdecomp = perdecomp
_dequant = dequantize


def _gradients(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Différences finies avant, périodiques."""
    return np.roll(u, -1, axis=1) - u, np.roll(u, -1, axis=0) - u


def _omega(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, -1.0, 1.0)
    return t * np.arcsin(t) + np.sqrt(1.0 - t * t) - 1.0


def sharpness_stats(image: np.ndarray, pmode: int = 3) -> dict:
    """
    Ingrédients de l'indice de netteté SI.

    On compare TV(u) à celle du champ gaussien V = u * W (convolution
    circulaire, W bruit blanc de variance 1/MN), de même spectre d'amplitude
    que u en moyenne mais de phase aléatoire. Avec g_a les gradients
    (a ∈ {x, y}) et Γ_ab(z) = Σ_w g_a(w) g_b(w+z) :

        μ  = E[TV(V)]   = sqrt(2/π) sqrt(MN) (‖g_x‖₂ + ‖g_y‖₂)
        σ² = Var[TV(V)] = (2/π) Σ_{a,b} sqrt(Γ_aa(0) Γ_bb(0)) Σ_z ω(ρ_ab(z)),
        ρ_ab = Γ_ab / sqrt(Γ_aa(0) Γ_bb(0)),  ω(t) = t arcsin t + sqrt(1-t²) - 1.

    Parameters
    ----------
    image : np.ndarray
        Image 2D en niveaux de gris.
    pmode : int {0, 1, 2, 3}, optional
        Prétraitement : 0 aucun ; 1 composante périodique ; 2 déquantification ;
        3 (défaut, recommandé) composante périodique puis déquantification.

    Returns
    -------
    dict
        ``"tv"``, ``"mu"``, ``"sigma"``, ``"z" = (μ - TV) / σ`` et ``"si"``.
    """
    if pmode not in (0, 1, 2, 3):
        raise ValueError(f"pmode doit valoir 0, 1, 2 ou 3 (reçu {pmode!r}).")
    u = _check_2d(image)
    if pmode in (1, 3):
        u, _ = _perdecomp(u)
    if pmode in (2, 3):
        u = _dequant(u)

    gx, gy = _gradients(u)
    tv = np.abs(gx).sum() + np.abs(gy).sum()
    MN = u.size
    mu = np.sqrt(2 / np.pi) * np.sqrt(MN) * (np.linalg.norm(gx) + np.linalg.norm(gy))

    F = [np.fft.fft2(gx), np.fft.fft2(gy)]
    norm2 = [np.sum(gx ** 2), np.sum(gy ** 2)]
    var = 0.0
    for a in range(2):
        for b in range(2):
            scale = np.sqrt(norm2[a] * norm2[b])
            if scale == 0:
                continue
            gamma = np.real(np.fft.ifft2(np.conj(F[a]) * F[b]))
            var += scale * _omega(gamma / scale).sum()
    sigma = np.sqrt(max(2 / np.pi * var, 0.0))

    if sigma == 0:  # image constante
        return {"tv": float(tv), "mu": float(mu), "sigma": 0.0, "z": 0.0, "si": 0.0}
    z = (mu - tv) / sigma
    si = -log_ndtr(-z) / np.log(10)  # -log10 P(N(0,1) > z), stable pour z grand
    return {"tv": float(tv), "mu": float(mu), "sigma": float(sigma), "z": float(z), "si": float(si)}


def sharpness_index(image: np.ndarray, pmode: int = 3) -> float:
    """
    Indice de netteté SI (Blanchet & Moisan, 2012) :

        SI(u) = -log10 Φ((μ - TV(u)) / σ),   Φ(t) = P(N(0,1) > t).

    Plus SI est grand, plus l'image est nette. Voir :func:`sharpness_stats`
    pour μ, σ, le détail du modèle et la signification de ``pmode``.

    Parameters
    ----------
    image : np.ndarray
        Image 2D en niveaux de gris.
    pmode : int {0, 1, 2, 3}, optional
        Prétraitement (défaut 3 : composante périodique puis
        déquantification ; à garder sauf image naturellement périodique ou
        non quantifiée).

    Returns
    -------
    float
        Valeur de l'indice (0 pour une image constante).
    """
    return sharpness_stats(image, pmode=pmode)["si"]
