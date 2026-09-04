"""
Mesure de netteté Antonel — métrique composite robuste aux variations
d'exposition, conçue pour l'imagerie satellite.

Implémentation clean-room d'après la description de :

    L. G. Antonel, "A Novel No-Reference Image Quality Metric for
    Assessing Sharpness in Satellite Imagery", arXiv:2410.10488, 2024.

Les étapes (dénoyautage, masquage d'intensité, sélection des gradients
forts par percentile, flou de référence, taux de décroissance normalisé
par axe) et les valeurs par défaut (percentiles 98.5/99.5, noyau de
référence 5×5 à σ=1) suivent le papier au plus près. Une limite connue :
l'indicateur de représentativité décrit en §3.6 du papier (calculé sur
une image floutée séparément avec un noyau 3× plus grand, σ=5, pour
détecter nuages/zones sans arête) n'est **pas** implémenté ici, faute de
code de référence ou d'exemple numérique pour en valider une
réimplémentation avec confiance — voir la note sur ``Rx``/``Ry``
ci-dessous.
"""
from __future__ import annotations

from typing import Dict, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

from .....core.image_utils import normalize_histogram


def antonel(
    image: npt.NDArray,
    pd: float = 98.5,
    pu: float = 99.5,
    ks: int = 5,
    kg: int = 5,
    sigma: float = 1.0,
    pdif: float = 0.15,
    return_details: bool = False,
) -> Union[float, Tuple[float, Dict[str, float]]]:
    """
    Calcule la mesure de netteté Antonel.

    Compare les gradients les plus intenses de l'image à ceux d'une version
    légèrement floutée de référence : le taux de décroissance normalisé
    donne un score robuste aux variations de bruit, d'exposition et de
    contenu de la scène. La décomposition en deux axes (horizontal/vertical)
    permet en outre de diagnostiquer un flou anisotrope (typiquement causé
    par un mouvement du capteur pendant la prise de vue).

    Étapes : (1) normalisation d'histogramme ; (2) dénoyautage (flou gaussien
    3×3 puis suppression des pixels dont la différence dépasse
    ``pdif`` × moyenne) ; (3) masquage des percentiles 1-99 d'intensité ;
    (4) gradients Sobel, sélection des gradients forts entre les percentiles
    ``pd``/``pu`` ; (5) gradients sur un flou de référence (``kg``, ``sigma``) ;
    (6) score = différence moyenne relative entre gradients forts et
    gradients de référence, en %.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    pd, pu : float, optional
        Percentiles bornant la sélection des gradients "forts".
        Par défaut ``98.5`` et ``99.5``.
    ks : int, optional
        Taille du noyau Sobel. Par défaut ``5``.
    kg : int, optional
        Taille du noyau du flou de référence. Par défaut ``5``.
    sigma : float, optional
        Écart-type du flou de référence. Par défaut ``1.0``.
    pdif : float, optional
        Seuil relatif de suppression d'anomalies. Par défaut ``0.15``.
    return_details : bool, optional
        Si ``True``, retourne aussi le détail directionnel du score.
        Par défaut ``False``.

    Returns
    -------
    float
        Score Antonel global (0-100 % ≈ net). Valeur plus élevée = image
        plus nette.
    Dict[str, float], optional
        ``{"score", "Sx", "Sy", "Rx", "Ry"}`` si ``return_details=True`` —
        ``Sx``/``Sy`` sont les composantes directionnelles du score.
        ``Rx``/``Ry`` sont la moyenne absolue des gradients forts par axe
        (utile pour du diagnostic) — **ce n'est pas** l'indicateur de
        représentativité du papier (§3.6), qui n'est pas implémenté ici
        (voir note en tête de module) ; ne pas l'utiliser comme un filtre
        de fiabilité du score.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = antonel(img)
    >>> score, details = antonel(img, return_details=True)
    """
    img = normalize_histogram(image).astype(np.float32)

    # 1. Dénoyautage / suppression d'anomalies
    blurred = cv2.GaussianBlur(img, (3, 3), 0.5)
    diff = np.abs(img - blurred)
    thresh = pdif * (np.mean(img) + 1e-6)
    img = np.where(diff > thresh, blurred, img)

    # 2. Masque d'intensité (exclure les outliers d'intensité)
    p_low, p_high = np.percentile(img, [1, 99])
    mask_lh = (img > p_low) & (img < p_high)

    # 3. Gradient Sobel
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ks)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ks)

    def _select_strong(g: npt.NDArray, mask: npt.NDArray) -> Tuple[npt.NDArray, float]:
        """Sélectionne les gradients forts compris entre les percentiles pd/pu."""
        g_m = g[mask]
        if g_m.size < 50:
            return np.zeros_like(g), 0.0
        low = np.percentile(np.abs(g_m), pd)
        high = np.percentile(np.abs(g_m), pu)
        strong_mask = mask & (np.abs(g) >= low) & (np.abs(g) <= high)
        return g * strong_mask, float(np.mean(np.abs(g[strong_mask]))) if np.any(strong_mask) else 0.0

    gx_s, rx = _select_strong(gx, mask_lh)
    gy_s, ry = _select_strong(gy, mask_lh)

    # 4. Gradient sur image légèrement floutée (référence)
    img_blur = cv2.GaussianBlur(img, (kg, kg), sigma)
    gx_b = cv2.Sobel(img_blur, cv2.CV_32F, 1, 0, ksize=ks)
    gy_b = cv2.Sobel(img_blur, cv2.CV_32F, 0, 1, ksize=ks)

    # 5. Score directionnel
    sx = sy = 0.0
    mask_x = np.abs(gx_s) > 0
    mask_y = np.abs(gy_s) > 0

    # Taux de décroissance par gradient puis moyenne (éq. 10-13 du papier :
    # Sx = 100 * mean(ΔGx_i), ΔGx_i = (Gx_i - GBx_i) / Gx_i) — et non un
    # ratio des moyennes, qui n'est pas équivalent en général.
    if np.any(mask_x):
        num = np.abs(gx_s[mask_x]) - np.abs(gx_b[mask_x])
        denom = np.abs(gx_s[mask_x]) + 1e-8
        sx = 100.0 * np.mean(num / denom)

    if np.any(mask_y):
        num = np.abs(gy_s[mask_y]) - np.abs(gy_b[mask_y])
        denom = np.abs(gy_s[mask_y]) + 1e-8
        sy = 100.0 * np.mean(num / denom)

    score = float(0.5 * (sx + sy))
    if not return_details:
        return score
    return score, {
        "score": score,
        "Sx": float(sx),
        "Sy": float(sy),
        "Rx": float(rx),
        "Ry": float(ry),
    }
