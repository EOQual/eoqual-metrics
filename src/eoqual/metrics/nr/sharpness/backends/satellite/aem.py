"""
AEM (Automatic Edge Method) — MTF calibrée sur arêtes naturelles auto-détectées.

Il ne s'agit pas d'une reproduction littérale d'un code publié, mais d'une
implémentation documentée respectant les principes physiques et critères de
sélection d'arêtes décrits dans la littérature remote sensing (slanted-edge
method, ISO 12233 ; sélection automatique d'arêtes naturelles pour la
caractérisation de MTF de capteurs d'observation de la Terre).
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt

from ._slanted_edge import detect_and_measure_edges


def aem(
    image: npt.NDArray,
    min_edge_length: int = 20,
    angle_range_deg: Tuple[float, float] = (2.0, 15.0),
    max_straightness_px: float = 1.0,
    half_width: int = 8,
    oversample: int = 4,
    min_valid_edges: int = 3,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, Union[float, int, bool]]]]:
    """
    Calcule la MTF@Nyquist par la méthode AEM (Automatic Edge Method).

    Standard du remote sensing pour caractériser et suivre la netteté d'un
    capteur d'observation de la Terre : contrairement à un score Brenner ou
    Tenengrad, la MTF est physiquement interprétable (contraste transmis à
    une fréquence spatiale donnée) et donc comparable entre capteurs, dates
    d'acquisition et missions.

    Méthode (slanted-edge, ISO 12233) appliquée sur des arêtes naturelles de
    la scène (bords de routes, limites de parcelles, bâtiments...), sans
    nécessiter de mire de calibration — voir
    :func:`~eoqual.metrics.nr.sharpness.backends.satellite._slanted_edge.detect_and_measure_edges`
    pour le détail des critères de sélection d'arête (longueur, angle,
    rectitude). Pour chaque arête retenue, la MTF@Nyquist, la MTF50 et la
    largeur à mi-hauteur (FWHM) sont mesurées ; le score final est la
    **médiane** de ces valeurs (robuste aux arêtes aberrantes isolées).

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    min_edge_length : int, optional
        Longueur minimale (px) d'un segment d'arête pour être candidat.
        Par défaut ``20``.
    angle_range_deg : tuple(float, float), optional
        Plage d'angle acceptée par rapport à l'axe horizontal/vertical le
        plus proche. Par défaut ``(2.0, 15.0)``.
    max_straightness_px : float, optional
        Écart-type perpendiculaire maximal accepté (rectitude).
        Par défaut ``1.0``.
    half_width : int, optional
        Demi-largeur (px) de la bande de projection autour de l'arête.
        Par défaut ``8``.
    oversample : int, optional
        Facteur de sur-échantillonnage de l'ESF (bins par pixel).
        Par défaut ``4``.
    min_valid_edges : int, optional
        Nombre minimal d'arêtes valides pour retourner un score non nul.
        Par défaut ``3``.
    return_details : bool, optional
        Si ``True``, retourne aussi le détail de la mesure (MTF50, FWHM,
        nombre d'arêtes). Par défaut ``False``.

    Returns
    -------
    float
        MTF@Nyquist médiane (0-1). Valeur plus élevée = image plus nette.
        ``0.0`` si moins de ``min_valid_edges`` arêtes exploitables.
    Dict[str, float or int or bool], optional
        ``{"score", "mtf_nyquist", "mtf50", "fwhm", "n_edges",
        "n_candidates", "insufficient"}`` si ``return_details=True``.
        ``insufficient=True`` signale un score non fiable (trop peu
        d'arêtes) — à filtrer en aval plutôt qu'à interpréter comme un flou
        réel : une scène homogène (eau, forêt dense) n'a simplement pas
        d'arête exploitable.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(512, 512).astype(np.float32)
    >>> score = aem(img)
    >>> score, details = aem(img, return_details=True)
    """
    measurements, n_candidates = detect_and_measure_edges(
        image, min_edge_length, angle_range_deg, max_straightness_px, half_width, oversample,
    )
    if len(measurements) < min_valid_edges:
        details = {
            "score": 0.0, "mtf_nyquist": 0.0, "mtf50": 0.0, "fwhm": 0.0,
            "n_edges": len(measurements), "n_candidates": n_candidates,
            "insufficient": True,
        }
        return (0.0, details) if return_details else 0.0

    mtf_nyq = np.array([m["mtf_nyquist"] for m in measurements])
    mtf50 = np.array([m["mtf50"] for m in measurements])
    fwhm = np.array([m["fwhm"] for m in measurements])
    score = float(np.median(mtf_nyq))
    if not return_details:
        return score
    return score, {
        "score": score,
        "mtf_nyquist": score,
        "mtf50": float(np.median(mtf50)),
        "fwhm": float(np.median(fwhm)),
        "n_edges": len(measurements),
        "n_candidates": n_candidates,
        "insufficient": False,
    }
