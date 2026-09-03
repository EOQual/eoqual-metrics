"""
PIQE — Perception based Image Quality Evaluator.

Implémentation propre, écrite à partir de la description publiée de
l'algorithme (voir Référence) — remplace l'implémentation précédemment
vendored (`backends/piqe/eadcat_impl.py`, source sans licence identifiée,
voir `THIRD_PARTY_LICENSES.md`). Algorithme entièrement déterministe (pas
de modèle appris), contrairement à NIQE/BRISQUE.

Principe
--------
1. Découper l'image (recadrée à un multiple de 16) en blocs 16×16 et
   calculer la transformée MSCN (même prétraitement que NIQE/BRISQUE,
   voir ``core/nss.py``).
2. Ne garder que les blocs "actifs" (variance MSCN > seuil) — les zones
   uniformes (ciel, fond flou) ne peuvent pas porter de distorsion
   perceptible.
3. Pour chaque bloc actif, deux critères indépendants : un **artefact de
   bloc** ("blocking" — le pourtour du bloc est découpé en segments de
   6 pixels ; un segment quasi constant, écart-type faible, trahit une
   transition artificielle entre blocs de compression) et un critère de
   **bruit** (écart-type du centre du bloc comparé à celui de son
   pourtour — un bloc où le centre est bien plus variable que son
   entourage, au-delà d'un facteur 2, est considéré bruité).
4. Score de distortion par bloc actif, agrégé avec un traitement
   particulier des 10% de blocs les moins dégradés (normalisation) puis
   moyenné sur le nombre de blocs actifs.

Référence
---------
Venkatanath, N., Praneeth, D., Bh, M. C., Channappayya, S. S., & Medasani,
S. S. (2015). *Blind Image Quality Evaluation Using Perception Based
Features*. National Conference on Communications (NCC), IEEE.
"""
from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

_BLOCK_SIZE = 16


def _mscn_transform(image: npt.NDArray) -> npt.NDArray:
    """
    Variante de la transformée MSCN utilisant un flou gaussien 2D direct
    (``cv2.GaussianBlur``, noyau 7×7, σ=7/6) plutôt que le noyau séparable
    de ``core.nss.mscn_transform`` — les deux discrétisent le même modèle
    gaussien de Mittal et al. à un ordre de précision près, mais PIQE
    (Venkatanath et al. 2015) a historiquement été validé avec cette
    variante ; conservée ici pour la fidélité numérique de la métrique.
    """
    img = image.astype(np.float32)
    mu = cv2.GaussianBlur(img, (7, 7), 7 / 6)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(img ** 2, (7, 7), 7 / 6) - mu ** 2))
    return (img - mu) / (sigma + 1.0)
_ACTIVITY_THRESHOLD = 0.1
_IMPAIRED_THRESHOLD = 0.1
_SEGMENT_SIZE = 6


def _edge_segments_min_std(block: npt.NDArray) -> float:
    """
    Écart-type minimal parmi tous les segments glissants de
    ``_SEGMENT_SIZE`` pixels sur les 4 bords du bloc — un minimum faible
    trahit un bord de compression (transition artificiellement plate).
    """
    edges = [block[0, :], block[:, -1], block[-1, :], block[:, 0]]
    n_segments = _BLOCK_SIZE - _SEGMENT_SIZE + 1
    min_std = np.inf
    for edge in edges:
        for start in range(n_segments):
            segment = edge[start:start + _SEGMENT_SIZE]
            min_std = min(min_std, float(np.std(segment)))
    return min_std


def _center_surround_ratio(block: npt.NDArray) -> float:
    """
    Ratio de l'écart-type du centre (2 colonnes centrales) sur celui du
    pourtour (colonnes restantes) du bloc.
    """
    center_cols = [block.shape[1] // 2 - 1, block.shape[1] // 2]
    center = block[:, center_cols]
    surround = np.delete(block, center_cols, axis=1)
    surround_std = float(np.std(surround))
    if surround_std == 0:
        return np.inf
    return float(np.std(center)) / surround_std


def piqe(P: npt.NDArray) -> float:
    """
    Calcule le score PIQE (Perception based Image Quality Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image en niveaux de gris (ou RGB, convertie en interne) de shape
        ``(H, W)`` ou ``(H, W, 3)``.

    Returns
    -------
    float
        Score PIQE, généralement dans ``[0, 100]``. Une valeur plus faible
        indique une meilleure qualité perçue. ``0.0`` si aucun bloc actif
        n'a été détecté (image quasi uniforme).

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256) * 255
    >>> score = piqe(img)
    """
    if P.ndim == 3:
        P = cv2.cvtColor(P.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    h, w = P.shape
    pad_h = (-h) % _BLOCK_SIZE
    pad_w = (-w) % _BLOCK_SIZE
    img = np.pad(P.astype(np.float64), ((0, pad_h), (0, pad_w)), mode="edge")

    mscn = _mscn_transform(img)

    block_scores = []
    n_active = 0
    for i in range(0, mscn.shape[0], _BLOCK_SIZE):
        for j in range(0, mscn.shape[1], _BLOCK_SIZE):
            block = mscn[i:i + _BLOCK_SIZE, j:j + _BLOCK_SIZE]
            block_var = float(np.var(block))
            if block_var <= _ACTIVITY_THRESHOLD:
                continue
            n_active += 1

            is_impaired = _edge_segments_min_std(block) < _IMPAIRED_THRESHOLD

            block_sigma = np.sqrt(block_var)
            center_surround = _center_surround_ratio(block)
            # Écart relatif entre l'écart-type du bloc et le ratio centre/
            # pourtour — pas le ratio lui-même (cf. Venkatanath et al. 2015).
            beta = abs(block_sigma - center_surround) / max(block_sigma, center_surround)
            is_noisy = block_sigma > 2 * beta

            score = (float(is_impaired) * (1 - block_var) ** 2
                     + float(is_noisy) * block_var ** 2)
            if score > 0:
                block_scores.append(score)

    if n_active == 0 or not block_scores:
        return 0.0

    block_scores.sort()
    n_low = max(1, int(0.1 * len(block_scores)))
    low_sum = sum(block_scores[:n_low])
    total = sum(block_scores)
    rescaled = [s * 10 * low_sum / total for s in block_scores] if total > 0 else block_scores

    C = 1.0
    return float((sum(rescaled) + C) / (C + n_active) * 100)
