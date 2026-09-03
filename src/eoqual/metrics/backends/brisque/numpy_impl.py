"""
BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator.

Implémentation propre, écrite à partir de la description publiée de
l'algorithme (voir Référence), réutilisant ``core/nss.py::aggd_fit`` pour
l'ajustement AGGD partagé avec NIQE — remplace l'implémentation
précédemment vendored (`backends/brisque/libsvm_impl/`, source sans
licence identifiée s'appuyant sur des binaires libsvm compilés vendored,
voir `THIRD_PARTY_LICENSES.md`).

Le modèle de régression (``models/allmodel``) est en revanche **conservé à
l'identique** — mêmes vecteurs de support que la release officielle
BRISQUE (Mittal, Moorthy, Bovik), sous licence de recherche permissive
avec citation obligatoire (voir ``models/NOTICE.md``). La prédiction SVR
(noyau RBF) est recalculée directement en numpy : plus besoin de
dépendre de libsvm ni de binaires compilés pour l'inférence — seul
l'entraînement d'un nouveau modèle nécessiterait cette bibliothèque.

Principe
--------
1. À deux échelles (image originale, puis sous-échantillonnée ×0,5), MSCN
   du niveau de gris normalisé dans ``[0, 1]``, puis AGGD des coefficients
   MSCN et de 4 produits de voisinage décalés (H, V, D1, D2) — 18 features
   par échelle, 36 au total (voir :func:`compute_features`).
2. Chaque feature est ramenée dans ``[-1, 1]`` par les bornes min/max
   apprises sur la base d'entraînement LIVE (constantes de la release
   officielle, voir :data:`_FEATURE_MIN`/:data:`_FEATURE_MAX`).
3. Score = prédiction d'un régresseur à vecteurs de support (SVR, noyau
   RBF) pré-entraîné sur des paires (features, note de qualité humaine).

Référence
---------
Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). *No-Reference Image
Quality Assessment in the Spatial Domain*. IEEE Transactions on Image
Processing, 21(12), 4695-4708.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List

import cv2
import numpy as np
import numpy.typing as npt

from ...core.nss import aggd_fit

_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "allmodel")

# Bornes min/max des 36 features, apprises sur la base d'entraînement LIVE
# (release officielle BRISQUE — voir models/NOTICE.md). Constantes de
# données, pas du code : nécessaires pour ramener chaque feature dans
# [-1, 1] avant de les soumettre au modèle SVR.
_FEATURE_MIN = np.array([
    0.336999, 0.019667, 0.230000, -0.125959, 0.000167, 0.000616,
    0.231000, -0.125873, 0.000165, 0.000600, 0.241000, -0.128814,
    0.000179, 0.000386, 0.243000, -0.133080, 0.000182, 0.000421,
    0.436998, 0.016929, 0.247000, -0.200231, 0.000104, 0.000834,
    0.257000, -0.200017, 0.000112, 0.000876, 0.257000, -0.155072,
    0.000112, 0.000356, 0.258000, -0.154374, 0.000117, 0.000351,
])
_FEATURE_MAX = np.array([
    9.999411, 0.807472, 1.644021, 0.202917, 0.712384, 0.468672,
    1.644021, 0.169548, 0.713132, 0.467896, 1.553016, 0.101368,
    0.687324, 0.533087, 1.554016, 0.101000, 0.689177, 0.533133,
    3.639918, 0.800955, 1.096995, 0.175286, 0.755547, 0.399270,
    1.095995, 0.155928, 0.751488, 0.402398, 1.041992, 0.093209,
    0.623516, 0.532925, 1.042992, 0.093714, 0.621958, 0.534484,
])

# Décalages (dy, dx) des 4 produits de voisinage H, V, D1 (diagonale),
# D2 (anti-diagonale) — mêmes orientations que ``core.nss.paired_products``,
# mais avec un remplissage à zéro plutôt que circulaire (convention de la
# release officielle BRISQUE, dont dépend le modèle pré-entraîné).
_SHIFTS = [(0, 1), (1, 0), (1, 1), (-1, 1)]


def _shift_zero_pad(image: npt.NDArray, dy: int, dx: int) -> npt.NDArray:
    """Décale ``image`` de ``(dy, dx)`` pixels, en comblant par des zéros."""
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image.astype(np.float32), M, (image.shape[1], image.shape[0]))


def _mscn_transform_01(image_uint8_scale: npt.NDArray) -> npt.NDArray:
    """
    MSCN sur image normalisée dans ``[0, 1]`` (convention BRISQUE — voir
    module). Mathématiquement équivalent à ``core.nss.mscn_transform`` sur
    l'image à l'échelle ``[0, 255]`` avec une constante additive de 1,
    mais reproduit ici la normalisation ``[0, 1]`` d'origine pour rester
    fidèle aux constantes ``_FEATURE_MIN``/``_FEATURE_MAX`` ci-dessus,
    apprises sur des features calculées à cette échelle précise.
    """
    im = image_uint8_scale.astype(np.float32) / 255.0
    mu = cv2.GaussianBlur(im, (7, 7), 7 / 6)
    sigma = np.sqrt(np.abs(cv2.GaussianBlur(im * im, (7, 7), 7 / 6) - mu ** 2))
    return (im - mu) / (sigma + 1.0 / 255.0)


def compute_features(image: npt.NDArray, n_scales: int = 2) -> npt.NDArray:
    """
    Calcule le vecteur de 36 features BRISQUE (2 échelles × 18 features).

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris, échelle ``[0, 255]``.
    n_scales : int, optional
        Nombre d'échelles (image originale + sous-échantillonnages ×0,5
        successifs). Par défaut ``2``.

    Returns
    -------
    npt.NDArray
        Vecteur de ``18 * n_scales`` features.
    """
    feats: List[float] = []
    current = image.astype(np.float32)

    for _ in range(n_scales):
        mscn = _mscn_transform_01(current)

        alpha0, left0, right0, _ = aggd_fit(mscn)
        feats.extend([alpha0, (left0 ** 2 + right0 ** 2) / 2.0])

        for dy, dx in _SHIFTS:
            product = _shift_zero_pad(mscn, dy, dx) * mscn
            alpha, left, right, mean_param = aggd_fit(product)
            feats.extend([alpha, mean_param, left ** 2, right ** 2])

        current = cv2.resize(current, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    return np.array(feats)


@dataclass
class _SVRModel:
    """Modèle epsilon-SVR à noyau RBF, tel que sérialisé par libsvm."""
    gamma: float
    rho: float
    coefficients: npt.NDArray  # shape (n_sv,)
    support_vectors: npt.NDArray  # shape (n_sv, n_features)

    def predict(self, x: npt.NDArray) -> float:
        """Prédiction epsilon-SVR : Σ(coef_i · K(x, sv_i)) - rho, noyau RBF."""
        sq_dist = np.sum((self.support_vectors - x) ** 2, axis=1)
        kernel = np.exp(-self.gamma * sq_dist)
        return float(np.dot(self.coefficients, kernel) - self.rho)


def _load_svr_model(path: str) -> _SVRModel:
    """
    Parse un fichier modèle libsvm (format texte standard) en
    :class:`_SVRModel`, sans dépendance à libsvm.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    gamma = None
    rho = None
    n_features = 0
    support_vectors: List[List[float]] = []
    coefficients: List[float] = []

    with open(path, "r") as f:
        in_sv_section = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "SV":
                in_sv_section = True
                continue
            if not in_sv_section:
                key, _, value = line.partition(" ")
                if key == "gamma":
                    gamma = float(value)
                elif key == "rho":
                    rho = float(value)
                continue

            tokens = line.split()
            coefficients.append(float(tokens[0]))
            sv = {}
            for tok in tokens[1:]:
                idx, _, val = tok.partition(":")
                sv[int(idx)] = float(val)
            n_features = max(n_features, max(sv))
            support_vectors.append(sv)

    if gamma is None or rho is None:
        raise ValueError(f"Fichier modèle invalide (gamma/rho manquant) : {path}")

    sv_matrix = np.zeros((len(support_vectors), n_features))
    for i, sv in enumerate(support_vectors):
        for idx, val in sv.items():
            sv_matrix[i, idx - 1] = val  # index libsvm 1-based

    return _SVRModel(
        gamma=gamma,
        rho=rho,
        coefficients=np.array(coefficients),
        support_vectors=sv_matrix,
    )


def brisque(P: npt.NDArray) -> float:
    """
    Calcule le score BRISQUE (Blind/Referenceless Image Spatial Quality
    Evaluator).

    Parameters
    ----------
    P : npt.NDArray
        Image en niveaux de gris (ou RGB, convertie en interne) de shape
        ``(H, W)`` ou ``(H, W, 3)``, échelle ``[0, 255]``.

    Returns
    -------
    float
        Score BRISQUE. Une valeur plus faible indique une meilleure
        qualité perçue.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256) * 255
    >>> score = brisque(img)
    """
    if P.ndim == 3:
        P = cv2.cvtColor(P.astype(np.uint8), cv2.COLOR_BGR2GRAY)

    features = compute_features(P)
    scaled = -1 + 2.0 * (features - _FEATURE_MIN) / (_FEATURE_MAX - _FEATURE_MIN)

    model = _load_svr_model(_MODEL_PATH)
    return model.predict(scaled)
