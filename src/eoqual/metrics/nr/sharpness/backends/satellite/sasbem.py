"""
SaSbEM (Semi-Automatic Statistically-Based Edge Method) — AEM + filtrage
statistique robuste des arêtes aberrantes.

Il ne s'agit pas d'une reproduction littérale d'un code publié, mais d'une
implémentation documentée respectant les principes physiques et critères de
sélection d'arêtes décrits dans la littérature remote sensing.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import numpy as np
import numpy.typing as npt

from ._slanted_edge import detect_and_measure_edges


def sasbem(
    image: npt.NDArray,
    min_edge_length: int = 20,
    angle_range_deg: Tuple[float, float] = (2.0, 15.0),
    max_straightness_px: float = 1.0,
    half_width: int = 8,
    oversample: int = 4,
    mad_k: float = 3.0,
    min_valid_edges: int = 3,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, Union[float, int, bool]]]]:
    """
    Calcule la MTF@Nyquist par la méthode SaSbEM (variante robuste d'AEM).

    Étend :func:`~eoqual.metrics.nr.sharpness.backends.satellite.aem.aem`
    (mêmes critères géométriques de sélection d'arêtes, voir
    :func:`~eoqual.metrics.nr.sharpness.backends.satellite._slanted_edge.detect_and_measure_edges`)
    avec une étape de robustesse statistique : après calcul de la MTF@Nyquist
    par arête, les arêtes dont le score s'écarte de la médiane de plus de
    ``mad_k`` fois la MAD (*Median Absolute Deviation*, robuste aux valeurs
    aberrantes contrairement à l'écart-type) sont exclues avant l'agrégation
    finale. Pensée pour exploiter un grand nombre d'arêtes naturelles
    (dizaines à centaines) sur de l'imagerie opérationnelle, où toutes les
    arêtes détectées ne sont pas de qualité égale (ombres, végétation en
    bordure, sur-saturation locale...).

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    min_edge_length, angle_range_deg, max_straightness_px, half_width, oversample
        Voir :func:`~eoqual.metrics.nr.sharpness.backends.satellite.aem.aem`.
    mad_k : float, optional
        Facteur multiplicatif de la MAD pour le filtrage des arêtes
        aberrantes (défaut ``3.0`` — équivalent à ~3σ pour une distribution
        gaussienne grâce au facteur de correction 1.4826).
    min_valid_edges : int, optional
        Nombre minimal d'arêtes valides (après filtrage) pour retourner un
        score non nul. Par défaut ``3``.
    return_details : bool, optional
        Si ``True``, retourne aussi le détail de la mesure (MTF50, FWHM,
        fiabilité). Par défaut ``False``.

    Returns
    -------
    float
        MTF@Nyquist médiane après filtrage MAD (0-1). Valeur plus élevée =
        image plus nette. ``0.0`` si moins de ``min_valid_edges`` arêtes
        exploitables (avant ou après filtrage).
    Dict[str, float or int or bool], optional
        ``{"score", "mtf_nyquist", "mtf50", "fwhm", "n_edges",
        "n_candidates", "reliability", "insufficient"}`` si
        ``return_details=True``. ``reliability`` est la proportion d'arêtes
        conservées après filtrage MAD par rapport au nombre total d'arêtes
        mesurées — proche de 1 signifie des arêtes cohérentes entre elles
        (bonne confiance dans le score) ; un indice faible signale une
        scène hétérogène en netteté (flou de bougé localisé, profondeur de
        champ variable, artefacts localisés).

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(512, 512).astype(np.float32)
    >>> score = sasbem(img)
    >>> score, details = sasbem(img, return_details=True)
    """
    measurements, n_candidates = detect_and_measure_edges(
        image, min_edge_length, angle_range_deg, max_straightness_px, half_width, oversample,
    )
    if len(measurements) < min_valid_edges:
        details = {
            "score": 0.0, "mtf_nyquist": 0.0, "mtf50": 0.0, "fwhm": 0.0,
            "n_edges": 0, "n_candidates": n_candidates, "reliability": 0.0,
            "insufficient": True,
        }
        return (0.0, details) if return_details else 0.0

    mtf_nyq = np.array([m["mtf_nyquist"] for m in measurements])
    med = np.median(mtf_nyq)
    mad = np.median(np.abs(mtf_nyq - med)) + 1e-12
    keep = np.abs(mtf_nyq - med) <= mad_k * 1.4826 * mad
    n_kept = int(keep.sum())

    if n_kept < min_valid_edges:
        details = {
            "score": 0.0, "mtf_nyquist": 0.0, "mtf50": 0.0, "fwhm": 0.0,
            "n_edges": n_kept, "n_candidates": n_candidates,
            "reliability": n_kept / len(measurements),
            "insufficient": True,
        }
        return (0.0, details) if return_details else 0.0

    mtf50 = np.array([m["mtf50"] for m in measurements])
    fwhm = np.array([m["fwhm"] for m in measurements])
    score = float(np.median(mtf_nyq[keep]))
    if not return_details:
        return score
    return score, {
        "score": score,
        "mtf_nyquist": score,
        "mtf50": float(np.median(mtf50[keep])),
        "fwhm": float(np.median(fwhm[keep])),
        "n_edges": n_kept,
        "n_candidates": n_candidates,
        "reliability": n_kept / len(measurements),
        "insufficient": False,
    }
