"""
Cœur commun à AEM et SaSbEM : détection automatique d'arêtes naturelles
(*slanted-edge*, ISO 12233) et mesure de la MTF par arête.

Module privé — non exposé dans l'API publique. Voir ``aem.py`` et
``sasbem.py`` pour les métriques qui l'utilisent.

Principe : repérer une arête nette légèrement inclinée par rapport aux axes
de l'image, projeter les intensités d'une bande de pixels perpendiculairement
à l'arête pour obtenir une Edge Spread Function (ESF) sur-échantillonnée
(l'angle non nul fournit l'échantillonnage sub-pixel), dériver pour obtenir
la Line Spread Function (LSF), puis calculer la FFT de la LSF pour obtenir la
courbe de MTF normalisée (1 à fréquence nulle).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from skimage.feature import canny
from skimage.measure import label, regionprops

from .....core.image_utils import normalize_histogram

__all__ = ["detect_and_measure_edges"]


def _extract_edge_segments(edge_map: NDArray[np.bool_], min_length: int) -> List[NDArray]:
    """Regroupe la carte binaire d'arêtes (Canny) en segments connexes (8-connexité)."""
    labeled = label(edge_map, connectivity=2)
    segments = []
    for region in regionprops(labeled):
        if region.area >= min_length:
            segments.append(region.coords.astype(np.float64))  # (row, col)
    return segments


def _line_angle_and_straightness(
    coords: NDArray,
) -> Tuple[float, float, float, NDArray, NDArray]:
    """
    Analyse en composantes principales (PCA) d'un segment d'arête.

    Returns
    -------
    angle_deg : orientation du segment par rapport à l'axe horizontal (0-180°).
    straightness_px : écart-type perpendiculaire à la direction principale
        (plus petit = arête plus rectiligne).
    length_px : étendue du segment le long de sa direction principale.
    main_dir : vecteur unitaire (drow, dcol) de la direction principale.
    perp_dir : vecteur unitaire perpendiculaire à ``main_dir``.
    """
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    main_dir = eigvecs[:, 0]
    perp_dir = np.array([-main_dir[1], main_dir[0]])

    angle_deg = float(np.degrees(np.arctan2(main_dir[1], main_dir[0])) % 180)
    straightness_px = float(np.sqrt(max(eigvals[1], 0.0)))

    proj = centered @ main_dir
    length_px = float(proj.max() - proj.min())

    return angle_deg, straightness_px, length_px, main_dir, perp_dir


def _project_edge_profile(
    image: NDArray[np.floating],
    coords: NDArray,
    main_dir: NDArray,
    perp_dir: NDArray,
    half_width: int,
    oversample: int,
) -> Optional[NDArray[np.floating]]:
    """
    Construit une Edge Spread Function (ESF) sur-échantillonnée en projetant
    tous les pixels d'une bande autour du segment d'arête sur l'axe
    perpendiculaire à celle-ci (voir docstring du module pour le principe).
    """
    rows, cols = image.shape
    r0, c0 = coords.mean(axis=0)

    r_min = max(0, int(coords[:, 0].min() - half_width))
    r_max = min(rows - 1, int(coords[:, 0].max() + half_width))
    c_min = max(0, int(coords[:, 1].min() - half_width))
    c_max = min(cols - 1, int(coords[:, 1].max() + half_width))

    rr, cc = np.meshgrid(np.arange(r_min, r_max + 1), np.arange(c_min, c_max + 1), indexing="ij")
    dr = rr - r0
    dc = cc - c0
    t_perp = dr * perp_dir[0] + dc * perp_dir[1]
    t_par = dr * main_dir[0] + dc * main_dir[1]

    par_centered = (coords - [r0, c0]) @ main_dir
    par_min, par_max = par_centered.min(), par_centered.max()
    band_mask = (np.abs(t_perp) <= half_width) & (t_par >= par_min) & (t_par <= par_max)

    if band_mask.sum() < 20:
        return None

    offsets = t_perp[band_mask]
    values = image[rr[band_mask], cc[band_mask]]

    bin_width = 1.0 / oversample
    bin_edges = np.arange(-half_width, half_width + bin_width, bin_width)
    bin_idx = np.clip(np.digitize(offsets, bin_edges) - 1, 0, len(bin_edges) - 2)

    esf = np.full(len(bin_edges) - 1, np.nan)
    for b in np.unique(bin_idx):
        esf[b] = values[bin_idx == b].mean()

    valid = ~np.isnan(esf)
    if valid.sum() < 0.5 * len(esf):
        return None
    esf = np.interp(np.arange(len(esf)), np.flatnonzero(valid), esf[valid])
    return esf


def _mtf_from_esf(esf: NDArray[np.floating], oversample: int) -> Optional[Dict[str, float]]:
    """LSF (dérivée de l'ESF) -> fenêtrage de Hann -> FFT -> courbe MTF normalisée (DC = 1)."""
    lsf = np.diff(esf)
    if np.abs(lsf).sum() < 1e-8:
        return None
    window = np.hanning(len(lsf))
    lsf_w = lsf * window
    lsf_w = lsf_w / (np.abs(lsf_w).sum() + 1e-12)

    n_fft = max(512, 8 * len(lsf_w))
    mtf_raw = np.abs(np.fft.rfft(lsf_w, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / oversample)  # cycles/pixel
    mtf_norm = mtf_raw / (mtf_raw[0] + 1e-12)

    mtf_nyquist = float(np.interp(0.5, freqs, mtf_norm))

    below_half = np.where(mtf_norm <= 0.5)[0]
    if len(below_half) > 0 and below_half[0] > 0:
        i1, i2 = below_half[0] - 1, below_half[0]
        f1, f2 = freqs[i1], freqs[i2]
        m1, m2 = mtf_norm[i1], mtf_norm[i2]
        mtf50 = f1 + (0.5 - m1) * (f2 - f1) / (m2 - m1 + 1e-12)
    else:
        mtf50 = 0.5

    half_max = np.abs(lsf_w).max() / 2
    above = np.where(np.abs(lsf_w) >= half_max)[0]
    fwhm_px = float((above[-1] - above[0] + 1) / oversample) if len(above) > 1 else float(len(lsf_w) / oversample)

    return {"mtf_nyquist": mtf_nyquist, "mtf50": float(mtf50), "fwhm": fwhm_px}


def detect_and_measure_edges(
    image: NDArray[np.floating],
    min_edge_length: int = 20,
    angle_range_deg: Tuple[float, float] = (2.0, 15.0),
    max_straightness_px: float = 1.0,
    half_width: int = 8,
    oversample: int = 4,
    canny_sigma: float = 1.5,
    low_threshold: float = 0.05,
    high_threshold: float = 0.2,
) -> Tuple[List[Dict[str, float]], int]:
    """
    Détecte des arêtes naturelles exploitables et mesure leur MTF individuelle.

    Cœur partagé par :func:`~eoqual.metrics.nr.sharpness.backends.satellite.aem.aem`
    et :func:`~eoqual.metrics.nr.sharpness.backends.satellite.sasbem.sasbem`.

    Critères de sélection d'une arête candidate (typiques de la littérature
    remote sensing) :

    * **longueur minimale** (``min_edge_length``) : arête assez longue pour
      un ESF fiable ;
    * **angle** par rapport à l'axe horizontal/vertical le plus proche
      compris dans ``angle_range_deg`` : évite les arêtes parfaitement
      alignées sur la grille de pixels (aucun échantillonnage sub-pixel
      possible) et les arêtes trop obliques (fenêtre de projection utile
      trop courte) ;
    * **rectitude** (``max_straightness_px``) : écart-type perpendiculaire à
      la direction principale (PCA) — élimine les arêtes courbes, bruitées,
      ou issues de plusieurs objets confondus par la détection.

    Parameters
    ----------
    image : NDArray
        Image 2-D, n'importe quel dtype numérique (normalisée en interne).
    min_edge_length : int, optional
        Longueur minimale (px) d'un segment d'arête pour être candidat.
    angle_range_deg : tuple(float, float), optional
        Plage d'angle acceptée par rapport à l'axe horizontal/vertical le
        plus proche. Par défaut ``(2.0, 15.0)``.
    max_straightness_px : float, optional
        Écart-type perpendiculaire maximal accepté (rectitude).
    half_width : int, optional
        Demi-largeur (px) de la bande de projection autour de l'arête.
    oversample : int, optional
        Facteur de sur-échantillonnage de l'ESF (bins par pixel).
    canny_sigma, low_threshold, high_threshold : float, optional
        Paramètres du détecteur de contours Canny (scikit-image).

    Returns
    -------
    measurements : List[Dict[str, float]]
        Une entrée ``{"mtf_nyquist", "mtf50", "fwhm"}`` par arête retenue.
    n_candidates : int
        Nombre total de segments d'arêtes détectés avant filtrage
        géométrique — utile pour diagnostiquer un taux de rejet élevé
        (souvent le signe d'une scène pauvre en structures linéaires
        plutôt qu'un défaut de netteté réel).
    """
    img = normalize_histogram(image).astype(np.float32)
    edge_map = canny(img, sigma=canny_sigma, low_threshold=low_threshold, high_threshold=high_threshold)
    segments = _extract_edge_segments(edge_map, min_length=min_edge_length)

    measurements: List[Dict[str, float]] = []

    for coords in segments:
        angle_deg, straightness_px, length_px, main_dir, perp_dir = _line_angle_and_straightness(coords)

        angle_mod = angle_deg % 90
        angle_from_axis = min(angle_mod, 90 - angle_mod)
        if not (angle_range_deg[0] <= angle_from_axis <= angle_range_deg[1]):
            continue
        if straightness_px > max_straightness_px:
            continue
        if length_px < min_edge_length:
            continue

        esf = _project_edge_profile(img, coords, main_dir, perp_dir, half_width, oversample)
        if esf is None:
            continue

        mtf_res = _mtf_from_esf(esf, oversample)
        if mtf_res is None:
            continue

        measurements.append(mtf_res)

    return measurements, len(segments)
