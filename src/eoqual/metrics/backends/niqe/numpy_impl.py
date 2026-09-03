"""
NIQE — Natural Image Quality Evaluator.

Implémentation propre, écrite à partir de la description publiée de
l'algorithme (voir Référence), réutilisant ``core/nss.py`` pour le
prétraitement MSCN/AGGD partagé avec BRISQUE — remplace l'implémentation
précédemment vendored (`backends/niqe/eadcat_impl.py`, sources sans licence
identifiée, voir `THIRD_PARTY_LICENSES.md`).

Les paramètres du modèle "pristine" (``models/modelparameters.mat``) sont
en revanche **conservés à l'identique** : ce fichier date du 24 août 2012
et correspond au fichier officiellement distribué par ses auteurs sur
``live.ece.utexas.edu`` (licence de recherche permissive avec citation
obligatoire — voir `THIRD_PARTY_LICENSES.md`), et non à une simple copie
non attribuée. Il ne peut pas être « réimplémenté » : c'est un jeu de
paramètres statistiques appris sur une base d'images naturelles, pas du
code.

Principe
--------
1. Découper l'image en blocs 96×96 (pleine résolution) et 48×48 (image
   sous-échantillonnée ×0,5).
2. Pour chaque bloc, calculer la transformée MSCN puis ajuster une AGGD
   aux coefficients MSCN et à leurs 4 produits de voisinage — 18 features
   par échelle, 36 au total.
3. Modéliser la distribution des features des blocs de l'image testée par
   une gaussienne multivariée (moyenne + covariance empiriques).
4. Score = distance de Mahalanobis entre ce modèle et le modèle "pristine"
   pré-appris (``pop_mu``/``pop_cov``) — plus la distance est grande, plus
   l'image s'écarte statistiquement des images naturelles nettes.

Référence
---------
Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). *Making a
"Completely Blind" Image Quality Analyzer*. IEEE Signal Processing
Letters, 20(3), 209-212.
"""
from __future__ import annotations

import os

import cv2
import numpy as np
import numpy.typing as npt
import scipy.io
import scipy.linalg
import scipy.special

from ...core.nss import aggd_fit, mscn_transform, paired_products

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "modelparameters.mat")
_PATCH_SIZE = 96


def _scaled_std(alpha: float, left: float, right: float) -> tuple:
    """
    Redimensionne les écarts-types bruts (gauche/droite) renvoyés par
    :func:`~eoqual.metrics.core.nss.aggd_fit` par le ratio AGGD
    ``√Γ(1/α) / √Γ(3/α)`` — c'est la forme utilisée par les *features*
    NIQE (contrairement à BRISQUE, qui utilise les écarts-types bruts).
    """
    ratio = np.sqrt(scipy.special.gamma(1.0 / alpha)) / np.sqrt(scipy.special.gamma(3.0 / alpha))
    return ratio * left, ratio * right


def _patch_features(patch: npt.NDArray) -> npt.NDArray:
    """
    Calcule les 18 features NIQE (AGGD du MSCN + 4 produits de voisinage)
    d'un bloc déjà transformé en coefficients MSCN.

    Note
    ----
    Pour les deux orientations diagonales (D1, D2), seule la composante
    ``bl`` (et non la paire ``bl, br``) est répétée deux fois dans le
    vecteur de features — convention de l'algorithme de référence, dont
    dépend directement le modèle "pristine" pré-appris
    (``models/modelparameters.mat``, voir le module). Utiliser ``bl, br``
    romprait la correspondance avec ce modèle et fausserait le score.
    """
    alpha0, left0, right0, _ = aggd_fit(patch)
    bl0, br0 = _scaled_std(alpha0, left0, right0)
    feats = [alpha0, (bl0 + br0) / 2.0]

    products = paired_products(patch)
    for idx, prod in enumerate(products):
        alpha, left, right, mean_param = aggd_fit(prod)
        bl, br = _scaled_std(alpha, left, right)
        if idx < 2:  # H, V : paire complète
            feats.extend([alpha, mean_param, bl, br])
        else:  # D1, D2 : bl répété (convention de l'algorithme de référence)
            feats.extend([alpha, mean_param, bl, bl])
    return np.array(feats)


def _extract_patch_features(image: npt.NDArray, patch_size: int) -> npt.NDArray:
    """Découpe ``image`` en blocs non chevauchants et calcule leurs features."""
    mscn, _, _ = mscn_transform(image)
    h, w = mscn.shape
    feats = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            block = mscn[j:j + patch_size, i:i + patch_size]
            feats.append(_patch_features(block))
    return np.array(feats)


def niqe(P: npt.NDArray, patch_size: int = _PATCH_SIZE) -> float:
    """
    Calcule le score NIQE (Natural Image Quality Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``, ``H`` et ``W`` >
        ``2 * patch_size + 1`` (résolution minimale ``193×193`` avec les
        paramètres par défaut — le modèle pristine a été appris à cette
        granularité de bloc).
    patch_size : int, optional
        Taille des blocs d'analyse (pleine résolution). Par défaut ``96``.

    Returns
    -------
    float
        Score NIQE. Une valeur plus faible indique une image
        statistiquement plus proche des images naturelles nettes.

    Raises
    ------
    ValueError
        Si l'image est trop petite pour la granularité de bloc choisie.
    FileNotFoundError
        Si le fichier de paramètres du modèle pristine est introuvable.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256) * 255
    >>> score = niqe(img)
    """
    if P.ndim == 3:
        P = cv2.cvtColor(P.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    h, w = P.shape
    if h <= 2 * patch_size + 1 or w <= 2 * patch_size + 1:
        raise ValueError(
            f"Image trop petite pour patch_size={patch_size} "
            f"(reçu {P.shape}, requis > {(2 * patch_size + 1, 2 * patch_size + 1)})."
        )

    if not os.path.isfile(_MODEL_PATH):
        raise FileNotFoundError(_MODEL_PATH)
    params = scipy.io.loadmat(_MODEL_PATH)
    pop_mu = np.ravel(params["mu_prisparam"])
    pop_cov = params["cov_prisparam"]

    # Recadrage pour un découpage exact en blocs, comme le fichier de
    # référence de l'algorithme.
    img = P.astype(np.float64)
    h_off, w_off = h % patch_size, w % patch_size
    if h_off:
        img = img[:-h_off, :]
    if w_off:
        img = img[:, :-w_off]

    img_half = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    feats_full = _extract_patch_features(img, patch_size)
    feats_half = _extract_patch_features(img_half, patch_size // 2)
    n = min(len(feats_full), len(feats_half))
    feats = np.hstack((feats_full[:n], feats_half[:n]))

    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)

    diff = sample_mu - pop_mu
    cov = (pop_cov + sample_cov) / 2.0
    pinv_cov = scipy.linalg.pinv(cov)
    return float(np.sqrt(diff @ pinv_cov @ diff))
