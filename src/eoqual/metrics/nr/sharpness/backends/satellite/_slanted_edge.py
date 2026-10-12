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
from scipy.ndimage import gaussian_filter1d
from skimage.feature import canny
from skimage.measure import label, regionprops

from eoqual.core import edge as core_edge

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
    return core_edge.bin_profile(offsets, values, half_width, oversample)


#: Rejet d'une arête dont la MTF dépasse ce seuil sur [0, Nyquist] : une MTF
#: physique ne dépasse pas 1 ; au-delà de la marge laissée au bruit, la LSF a
#: des lobes négatifs importants (arête voisine dans la bande, coin, texture)
#: et la MTF@Nyquist mesurée est biaisée vers le haut.
MAX_MTF_OVERSHOOT = 1.5
#: Rejet d'une arête dont |Σ LSF| / Σ |LSF| est sous ce seuil : la LSF change
#: de signe au point de s'annuler presque (ligne fine plutôt que bord) et la
#: normalisation en fréquence nulle divise par ~0.
MIN_EDGE_SIGN_RATIO = 0.1


def _mtf_from_esf(
    esf: NDArray[np.floating],
    oversample: int,
    max_mtf_overshoot: float = MAX_MTF_OVERSHOOT,
    min_edge_sign_ratio: float = MIN_EDGE_SIGN_RATIO,
) -> Optional[Dict[str, float]]:
    """
    LSF (dérivée de l'ESF) -> fenêtrage de Hann -> FFT zéro-paddée -> MTF
    normalisée en fréquence nulle, puis MTF@Nyquist, MTF50 et FWHM.

    Calcul de MTF et grandeurs dérivées délégués à :mod:`eoqual.core.edge`
    (partagés avec eoqual-measure).

    FWHM : calculée sur la LSF lissée par une gaussienne d'un demi-pixel
    d'écart-type (``oversample / 2`` échantillons), puis par la première
    traversée du demi-maximum de part et d'autre du pic. Le lissage est
    nécessaire : la dérivée d'une ESF sur-échantillonnée par binning
    présente des dents de scie (bins vides interpolés, bruit), dont un pic
    isolé ramènerait la FWHM à un seul échantillon (~0.3 px). Il élargit la
    FWHM d'environ 0.1-0.2 px pour des flous usuels. L'ancienne définition
    (premier et dernier point au-dessus du demi-maximum sur tout le profil)
    surestimait au contraire la largeur dès qu'un rebond de bruit, même
    loin du pic, repassait au-dessus du seuil (jusqu'à 15 px observés).

    Renvoie ``None`` (arête rejetée) si la LSF s'annule presque
    (``|Σ LSF| / Σ |LSF| < min_edge_sign_ratio``) ou si la MTF dépasse
    ``max_mtf_overshoot`` avant Nyquist — voir :data:`MAX_MTF_OVERSHOOT` et
    :data:`MIN_EDGE_SIGN_RATIO`.
    """
    lsf = np.diff(esf)
    if np.abs(lsf).sum() < 1e-8:
        return None
    lsf_w = lsf * np.hanning(len(lsf))
    lsf_w = lsf_w / (np.abs(lsf_w).sum() + 1e-12)
    if abs(lsf_w.sum()) < min_edge_sign_ratio:
        return None  # ligne fine ou profil sans saut net : pas un bord

    dx = 1.0 / oversample
    freqs, mtf = core_edge.lsf_to_mtf(lsf_w, dx, n_fft=max(512, 8 * len(lsf_w)))  # cycles/pixel
    if np.max(mtf[freqs <= 0.5]) > max_mtf_overshoot:
        return None  # MTF non physique : arête contaminée

    mtf_nyquist = core_edge.mtf_at(freqs, mtf, 0.5)
    mtf50 = core_edge.frequency_at(freqs, mtf, 0.5)
    if np.isnan(mtf50):
        mtf50 = 0.5  # MTF jamais sous 0.5 avant Nyquist : plafonnée à Nyquist

    x = np.arange(len(lsf_w)) * dx
    lsf_smooth = gaussian_filter1d(lsf_w, oversample / 2.0)
    fwhm_px = core_edge.fwhm(x, np.abs(lsf_smooth))
    if fwhm_px == 0.0:  # demi-maximum non retraversé d'un côté (pic en bord de profil)
        fwhm_px = float(len(lsf_w) * dx)

    return {"mtf_nyquist": float(mtf_nyquist), "mtf50": float(mtf50), "fwhm": float(fwhm_px)}


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
    max_mtf_overshoot: float = MAX_MTF_OVERSHOOT,
    min_edge_sign_ratio: float = MIN_EDGE_SIGN_RATIO,
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
    max_mtf_overshoot : float, optional
        Rejette une arête dont la MTF dépasse ce seuil sur [0, Nyquist]
        (défaut :data:`MAX_MTF_OVERSHOOT` = 1.5).
    min_edge_sign_ratio : float, optional
        Rejette une arête dont ``|Σ LSF| / Σ |LSF|`` est inférieur à ce
        seuil — ligne fine plutôt que bord (défaut
        :data:`MIN_EDGE_SIGN_RATIO` = 0.1).

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

        mtf_res = _mtf_from_esf(esf, oversample, max_mtf_overshoot, min_edge_sign_ratio)
        if mtf_res is None:
            continue

        measurements.append(mtf_res)

    return measurements, len(segments)
