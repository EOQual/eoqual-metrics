"""
Estimation de la MTF (Modulation Transfer Function) par arêtes ponctuelles.

Contrairement aux métriques génériques de netteté (Tenengrad, Laplacien...),
la MTF est LA métrique standard utilisée par les opérateurs de satellites
d'observation de la Terre pour caractériser et suivre la netteté de leurs
capteurs : elle est physiquement interprétable (contraste transmis à une
fréquence spatiale donnée, en cycles/pixel) et donc comparable entre
capteurs, dates d'acquisition et missions — contrairement à un score qui
dépend fortement du contenu de la scène.

Principe (slanted-edge method, ISO 12233) : détection d'arêtes (Canny),
extraction d'un profil d'intensité le long de la normale locale au gradient
en chaque pixel d'arête retenu, moyenne de ces profils pour obtenir une Line
Spread Function (LSF), puis FFT de la LSF pour obtenir la courbe de MTF.

Note
----
Cette implémentation est volontairement légère : arêtes ponctuelles
(direction locale du gradient), sans contrainte d'angle ni de rectitude
géométrique sur des segments d'arête complets. C'est un indicateur rapide,
à ne pas confondre avec les méthodes ``aem``/``sasbem``
(:mod:`eoqual.metrics.nr.sharpness.backends.satellite`), plus rigoureuses
(sélection géométrique d'arêtes complètes par PCA, filtrage statistique
pour ``sasbem``) mais plus coûteuses — voir leur documentation pour la
méthodologie complète et les références bibliographiques.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from skimage.feature import canny

from .....core.image_utils import normalize_histogram


def mtf(
    image: npt.NDArray,
    max_kernel_size: int = 15,
    n_edges: int = 150,
    seed: int = 0,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, Union[float, int, bool]]]]:
    """
    Estime la MTF@Nyquist d'une image par arêtes ponctuelles (Canny + FFT).

    1. Détection d'arêtes avec Canny (sigma = 1.5) et sélection des pixels
       d'arête dont le gradient dépasse le 90ᵉ percentile.
    2. Sous-échantillonnage déterministe (``seed``) de ``n_edges`` pixels
       d'arête parmi les candidats.
    3. Pour chaque pixel retenu, profil d'intensité le long de la direction
       locale du gradient (≈ ``max_kernel_size`` pixels), transformé en
       *line-spread function* (LSF) par différence absolue successive.
    4. Moyenne de tous les LSF, puis FFT pour obtenir la courbe de MTF
       normalisée (1 à fréquence nulle).
    5. Lecture de la MTF à 0,5 cycle/pixel (Nyquist) et de la fréquence à
       laquelle elle descend à 0,5 (MTF50).

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    max_kernel_size : int, optional
        Demi-largeur (×2+1) de la fenêtre de profil autour de chaque pixel
        d'arête. Par défaut ``15``.
    n_edges : int, optional
        Nombre maximal de pixels d'arête échantillonnés. Par défaut ``150``.
    seed : int, optional
        Graine du générateur aléatoire pour le sous-échantillonnage
        (reproductibilité — appeler cette fonction deux fois sur la même
        image renvoie exactement le même résultat). Par défaut ``0``.
    return_details : bool, optional
        Si ``True``, retourne aussi le détail de la mesure (MTF50, FWHM,
        nombre d'arêtes). Par défaut ``False``.

    Returns
    -------
    float
        MTF@Nyquist (0-1). Valeur plus élevée = image plus nette.
        ``0.0`` si trop peu d'arêtes exploitables ont été détectées.
    Dict[str, float or int or bool], optional
        ``{"score", "mtf_nyquist", "mtf50", "fwhm", "n_edges",
        "insufficient"}`` si ``return_details=True``. ``insufficient=True``
        signale un score non fiable (trop peu d'arêtes) plutôt qu'un flou
        réel — une scène homogène (eau, forêt dense) n'a simplement pas
        d'arête exploitable.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = mtf(img)
    >>> score, details = mtf(img, return_details=True)
    """
    img = normalize_histogram(image).astype(np.float32)
    edges = canny(img, sigma=1.5, low_threshold=0.05, high_threshold=0.2)
    gy, gx = np.gradient(img)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    edge_coords = np.column_stack(np.where(edges & (magnitude > np.percentile(magnitude, 90))))

    def _insufficient(fwhm: float, n: int) -> Union[float, Tuple[float, Dict]]:
        details = {
            "score": 0.0, "mtf_nyquist": 0.0, "mtf50": 0.0, "fwhm": fwhm,
            "n_edges": n, "insufficient": True,
        }
        return (0.0, details) if return_details else 0.0

    if len(edge_coords) < 30:
        return _insufficient(10.0, 0)

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(edge_coords), min(n_edges, len(edge_coords)), replace=False)
    selected = edge_coords[idx]
    half = max_kernel_size // 2
    profiles: List[npt.NDArray] = []
    for y, x in selected:
        dx, dy = gx[y, x], gy[y, x]
        norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-8
        dx, dy = dx / norm, dy / norm
        vals: List[float] = []
        for t in range(-half, half + 1):
            yy = int(round(y + t * dy))
            xx = int(round(x + t * dx))
            if 0 <= yy < img.shape[0] and 0 <= xx < img.shape[1]:
                vals.append(img[yy, xx])
        if len(vals) < max_kernel_size // 2:
            continue
        vals_arr = np.array(vals)
        lsf = np.abs(np.diff(vals_arr))
        if lsf.max() < 1e-6:
            continue
        lsf = lsf / (lsf.sum() + 1e-12)
        profiles.append(lsf)

    if len(profiles) < 8:
        return _insufficient(8.0, len(profiles))

    min_len = min(len(p) for p in profiles)
    kernel = np.mean([p[:min_len] for p in profiles], axis=0)
    kernel = kernel / (kernel.sum() + 1e-12)

    # FFT du LSF moyen -> courbe MTF normalisée (DC = 1)
    n_fft = max(256, 8 * len(kernel))
    mtf_raw = np.abs(np.fft.rfft(kernel, n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, d=1.0)  # cycles/pixel, jusqu'à Nyquist = 0.5
    mtf_norm = mtf_raw / (mtf_raw[0] + 1e-12)
    mtf_nyquist = float(np.interp(0.5, freqs, mtf_norm))

    below_half = np.where(mtf_norm <= 0.5)[0]
    if len(below_half) > 0 and below_half[0] > 0:
        i1, i2 = below_half[0] - 1, below_half[0]
        f1, f2 = freqs[i1], freqs[i2]
        m1, m2 = mtf_norm[i1], mtf_norm[i2]
        mtf50 = f1 + (0.5 - m1) * (f2 - f1) / (m2 - m1 + 1e-12)
    else:
        mtf50 = 0.5  # la MTF ne descend pas sous 0.5 avant Nyquist -> image très nette

    half_max = kernel.max() / 2
    above = np.where(kernel >= half_max)[0]
    fwhm = float(above[-1] - above[0] + 1) if len(above) > 1 else float(max_kernel_size)

    if not return_details:
        return mtf_nyquist
    return mtf_nyquist, {
        "score": mtf_nyquist,
        "mtf_nyquist": mtf_nyquist,
        "mtf50": float(mtf50),
        "fwhm": fwhm,
        "n_edges": len(profiles),
        "insufficient": False,
    }
