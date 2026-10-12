#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ISSM (Information theoretic-based Statistic Similarity Measure).

M. A. Aljanabi, Z. M. Hussain, N. A. A. Shnain, S. F. Lu, "Design of a
hybrid measure for image similarity: a statistical, algebraic, and
information-theoretic approach", European Journal of Remote Sensing,
52(sup4), pp. 2-15, 2019. doi:10.1080/22797254.2019.1628617

Implémentation originale de l'auteur à partir de l'article (MIT). Équation
(16) de l'article :

    I(x, y) = (C·EHS·(a + b) + e) / (a·C·EHS + b·EHS + c·S + e)

avec a, b, c = 0.3, 0.5, 0.7 ; C la corrélation (en valeur absolue) des
cartes de contours de Canny (éq. 14) ; EHS l'entropie du histogramme
conjoint (éq. 13) ; S le SSIM (éq. 1, L = 255).

Choix là où l'article est muet ou incohérent (voir REFERENCE_TECHNIQUE.md) :

- **EHS** : les équations (10)-(13) normalisent le histogramme conjoint
  (probabilités), mais avec cette lecture deux images identiques donnent
  ~0.92 et deux images sans rapport ~0.3, ce que contredisent les figures
  3 à 10 de l'article (→ 1 et ≈ 0). Les figures ne sont reproduites
  qu'avec un histogramme en **comptes** : c'est le défaut
  (``ehs="counts"``) ; ``ehs="probabilities"`` suit les équations à la
  lettre.
- **e** : non défini dans l'article ; nombre d'Euler (comme
  image-similarity-measures).
- **Canny** : seuils non donnés ; les auteurs travaillent sous MATLAB,
  dont ``edge(I, 'canny')`` choisit les seuils automatiquement. Reproduit
  ici : σ = √2, seuil haut = 70e centile de la norme du gradient (70 % de
  pixels non-contours), seuil bas = 0.4 × seuil haut.
- **Dynamique** : l'article suppose des images 8 bits (L = 255) ; les
  images sont ramenées sur [0, 255] à partir de ``data_range``.

Conséquences de la lecture « comptes » : EHS est négative et de grande
amplitude (elle croît comme N log N), si bien que
I ≈ 0.8·C / (0.3·C + 0.5) : l'ISSM est dominé par la corrélation des
contours, le SSIM ne pesant presque rien. Pour deux images identiques,
I = 1 + 0.7 / (0.8 |EHS|), soit 1 à 4e-7 près sur 512 × 512.
"""
from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy import ndimage
from skimage.feature import canny
from skimage.metrics import structural_similarity

__all__ = ["issm"]

A, B, C = 0.3, 0.5, 0.7
CANNY_SIGMA = math.sqrt(2.0)
CANNY_NON_EDGE_FRACTION = 0.7
CANNY_LOW_RATIO = 0.4


def _to_8bit(image: npt.NDArray, low: float, data_range: float) -> npt.NDArray[np.float64]:
    return np.clip(np.round((image - low) / data_range * 255.0), 0.0, 255.0)


def _canny_matlab(image: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
    """Canny à seuils automatiques, à la manière de MATLAB ``edge(I, 'canny')``."""
    smoothed = ndimage.gaussian_filter(image, CANNY_SIGMA)
    magnitude = np.hypot(ndimage.sobel(smoothed, axis=0), ndimage.sobel(smoothed, axis=1))
    high = float(np.quantile(magnitude, CANNY_NON_EDGE_FRACTION))
    if high <= 0:
        return np.zeros(image.shape, dtype=bool)
    return canny(image, sigma=CANNY_SIGMA, low_threshold=CANNY_LOW_RATIO * high, high_threshold=high)


def _edge_correlation(x8: npt.NDArray[np.float64], y8: npt.NDArray[np.float64]) -> float:
    """Éq. (14) : |corrélation de Pearson| des cartes de contours."""
    g = _canny_matlab(x8).astype(np.float64)
    h = _canny_matlab(y8).astype(np.float64)
    gc, hc = g - g.mean(), h - h.mean()
    denominator = math.sqrt(float(np.sum(gc**2) * np.sum(hc**2)))
    if denominator == 0.0:  # au moins une carte sans contour (image uniforme)
        return 1.0 if np.array_equal(g, h) else 0.0
    return abs(float(np.sum(gc * hc)) / denominator)


def _ehs(x8: npt.NDArray[np.float64], y8: npt.NDArray[np.float64], mode: str) -> float:
    """Éq. (13) : entropie du histogramme conjoint (256 × 256 classes)."""
    joint = np.histogram2d(x8.ravel(), y8.ravel(), bins=256, range=[[0, 256], [0, 256]])[0]
    t = joint[joint > 0]
    if mode == "probabilities":
        t = t / x8.size
    elif mode != "counts":
        raise ValueError(f"ehs doit valoir 'counts' ou 'probabilities' (reçu {mode!r}).")
    return float(-np.sum(t * np.log2(t)))


def issm(
    GT: npt.NDArray,
    P: npt.NDArray,
    data_range: float | None = None,
    ehs: str = "counts",
) -> float:
    """
    ISSM entre une image de référence et une image dégradée (niveaux de gris).

    Parameters
    ----------
    GT, P :
        Images 2D de même forme.
    data_range :
        Dynamique des images (ex. 255, 4095, 1.0). Défaut : ``max(GT) -
        min(GT)``. Les deux images sont ramenées sur [0, 255] avec
        l'origine ``min(GT)``.
    ehs :
        ``"counts"`` (défaut, reproduit les figures de l'article) ou
        ``"probabilities"`` (équations 10-13 à la lettre) — voir l'en-tête
        du module.

    Returns
    -------
    float
        Similarité ; 1 pour deux images identiques (``ehs="counts"``),
        proche de 0 pour deux images sans rapport.
    """
    gt = np.asarray(GT, dtype=np.float64)
    p = np.asarray(P, dtype=np.float64)
    if gt.ndim != 2:
        raise ValueError(f"ISSM est défini sur des images en niveaux de gris (2D), reçu shape={gt.shape}.")
    low = float(gt.min())
    if data_range is None:
        data_range = float(gt.max() - low)
    if data_range <= 0:
        data_range = 1.0
    x8, y8 = _to_8bit(gt, low, data_range), _to_8bit(p, low, data_range)

    c_val = _edge_correlation(x8, y8)
    e_val = _ehs(x8, y8, ehs)
    s_val = float(structural_similarity(x8, y8, data_range=255))
    numerator = c_val * e_val * (A + B) + math.e
    denominator = A * c_val * e_val + B * e_val + C * s_val + math.e
    return float(numerator / denominator)
