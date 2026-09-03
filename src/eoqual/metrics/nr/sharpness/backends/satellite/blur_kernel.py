"""
Estimation d'un noyau de flou (PSF) par profils d'arêtes.
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
from skimage.feature import canny

from .....core.image_utils import normalize_histogram


def blur_kernel(
    image: npt.NDArray,
    max_kernel_size: int = 15,
    n_edges: int = 150,
    seed: int = 0,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Estime un noyau de flou (*point-spread function*, PSF) à partir d'arêtes fortes.

    1. Détection d'arêtes avec Canny (sigma = 1.5).
    2. Sélection des arêtes dont le gradient est supérieur au 90ᵉ percentile.
    3. Pour chaque arête sélectionnée, profil d'intensité le long de la
       normale locale (≈ ``max_kernel_size`` pixels).
    4. Le profil est transformé en *line-spread function* (LSF) — différence
       absolue entre intensités successives, normalisée.
    5. Tous les LSF sont moyennés pour obtenir le noyau estimé.
    6. **score** : norme L2 du noyau (plus grand → moins de flou).
    7. **fwhm** : largeur à mi-hauteur du noyau (plus petite → image nette).

    Le sous-échantillonnage des arêtes (``n_edges`` parmi les candidates) est
    déterministe (``seed``) : appeler cette fonction deux fois sur la même
    image renvoie exactement le même résultat.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    max_kernel_size : int, optional
        Demi-largeur (×2+1) de la fenêtre de profil autour de chaque arête.
        Par défaut ``15``.
    n_edges : int, optional
        Nombre maximal d'arêtes échantillonnées. Par défaut ``150``.
    seed : int, optional
        Graine du générateur aléatoire pour le sous-échantillonnage des
        arêtes (reproductibilité). Par défaut ``0``.
    return_details : bool, optional
        Si ``True``, retourne aussi la largeur à mi-hauteur (``fwhm``).
        Par défaut ``False``.

    Returns
    -------
    float
        Norme L2 du noyau de flou estimé. Valeur plus élevée = image plus nette.
        ``0.0`` si trop peu d'arêtes exploitables ont été détectées.
    Dict[str, float], optional
        ``{"score", "fwhm"}`` si ``return_details=True``.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = blur_kernel(img)
    """
    img = normalize_histogram(image).astype(np.float32)
    edges = canny(img, sigma=1.5, low_threshold=0.05, high_threshold=0.2)

    gy, gx = np.gradient(img)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    edge_coords = np.column_stack(
        np.where(edges & (magnitude > np.percentile(magnitude, 90)))
    )
    if len(edge_coords) < 30:
        return (0.0, {"score": 0.0, "fwhm": 10.0}) if return_details else 0.0

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
        return (0.0, {"score": 0.0, "fwhm": 8.0}) if return_details else 0.0

    min_len = min(len(p) for p in profiles)
    kernel = np.mean([p[:min_len] for p in profiles], axis=0)
    kernel = kernel / (kernel.sum() + 1e-12)

    half_max = kernel.max() / 2
    above = np.where(kernel >= half_max)[0]
    fwhm = float(above[-1] - above[0] + 1) if len(above) > 1 else float(max_kernel_size)

    score = float(np.linalg.norm(kernel))
    if not return_details:
        return score
    return score, {"score": score, "fwhm": fwhm}
