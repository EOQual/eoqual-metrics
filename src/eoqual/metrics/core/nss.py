"""
Statistiques de scène naturelle (Natural Scene Statistics) — briques
partagées par les métriques ``niqe`` et ``brisque``.

Implémentation propre, écrite à partir de la description publiée des deux
algorithmes (voir Références) — remplace les implémentations précédemment
vendored (`backends/niqe/eadcat_impl.py`, `backends/brisque/libsvm_impl/brisque.py`,
voir `THIRD_PARTY_LICENSES.md`). Les deux métriques reposent sur le même
prétraitement (transformée MSCN) et le même ajustement de distribution
(AGGD/GGD) — factoriser ce cœur commun évite de dupliquer deux fois une
mathématique identique.

Principe (commun aux deux métriques)
-------------------------------------
1. **MSCN** (Mean Subtracted Contrast Normalized) : soustraction d'une
   moyenne locale (flou gaussien 7×7, σ=7/6) et division par un écart-type
   local, avec une constante additive au dénominateur (stabilité
   numérique). Ce prétraitement rend l'image quasi-gaussienne pour une
   scène "naturelle" ; toute distorsion (flou, bruit, blocking...) dévie
   cette statistique de façon caractéristique.
2. **AGGD** (Asymmetric Generalized Gaussian Distribution) : les
   coefficients MSCN (et leurs produits de voisinage H/V/diagonales, qui
   capturent les corrélations directionnelles) sont ajustés à une AGGD à
   trois paramètres (forme, écart-type gauche, écart-type droit) par la
   méthode des moments (Lasmar et al.), résolue par recherche du meilleur
   paramètre de forme sur une grille précalculée.

Références
----------
Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). *Making a
"Completely Blind" Image Quality Analyzer*. IEEE Signal Processing
Letters, 20(3), 209-212.

Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). *No-Reference Image
Quality Assessment in the Spatial Domain*. IEEE Transactions on Image
Processing, 21(12), 4695-4708.

Lasmar, N.-E., Stitou, Y., & Berthoumieu, Y. (2009). *Multiscale skewed
heavy tailed model for texture analysis*. ICIP 2009 (estimation AGGD par
méthode des moments, utilisée par les deux papiers ci-dessus).
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy.ndimage import correlate1d
from scipy.special import gamma as _gamma_fn

# Grille de recherche du paramètre de forme AGGD/GGD et table précalculée
# de r(γ) = Γ(2/γ)² / (Γ(1/γ)·Γ(3/γ)), utilisée pour estimer γ par
# correspondance de moments (évite une optimisation itérative par image).
_GAMMA_RANGE = np.arange(0.2, 10.0, 0.001)
_R_TABLE = (_gamma_fn(2.0 / _GAMMA_RANGE) ** 2) / (
    _gamma_fn(1.0 / _GAMMA_RANGE) * _gamma_fn(3.0 / _GAMMA_RANGE)
)


def _gaussian_window(half_width: int, sigma: float) -> npt.NDArray:
    """
    Noyau gaussien 1D discret et normalisé, de longueur ``2*half_width+1``.
    """
    i = np.arange(-half_width, half_width + 1, dtype=np.float64)
    w = np.exp(-0.5 * (i ** 2) / (sigma ** 2))
    return w / w.sum()


# Noyau canonique utilisé par NIQE et BRISQUE : demi-largeur 3 (7 taps),
# σ = 7/6 — paramètres originaux de Mittal et al.
_GAUSS_WINDOW = _gaussian_window(3, 7.0 / 6.0)


def mscn_transform(
    image: npt.NDArray,
    window: npt.NDArray = _GAUSS_WINDOW,
    constant: float = 1.0,
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Calcule la transformée MSCN (Mean Subtracted Contrast Normalized).

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris, shape ``(H, W)``, échelle quelconque
        (ex. ``[0, 255]``) — non renormalisée en interne.
    window : npt.NDArray, optional
        Noyau 1D séparable pour l'estimation de la moyenne/variance
        locale. Par défaut le noyau gaussien canonique (7 taps, σ=7/6).
    constant : float, optional
        Constante additive au dénominateur, évite la division par une
        variance locale nulle (zones uniformes). Par défaut ``1.0``
        (échelle ``[0, 255]`` — utiliser une valeur proportionnellement
        plus petite si l'image est déjà normalisée dans ``[0, 1]``).

    Returns
    -------
    mscn : npt.NDArray
        Coefficients MSCN, même shape que ``image``.
    mu : npt.NDArray
        Moyenne locale.
    sigma : npt.NDArray
        Écart-type local.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(64, 64) * 255
    >>> mscn, mu, sigma = mscn_transform(img)
    """
    img = image.astype(np.float64)
    mu = correlate1d(img, window, axis=0, mode="constant")
    mu = correlate1d(mu, window, axis=1, mode="constant")
    var = correlate1d(img ** 2, window, axis=0, mode="constant")
    var = correlate1d(var, window, axis=1, mode="constant")
    sigma = np.sqrt(np.abs(var - mu ** 2))
    mscn = (img - mu) / (sigma + constant)
    return mscn, mu, sigma


def paired_products(mscn: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Produits de voisinage des coefficients MSCN dans 4 directions.

    Capturent les corrélations directionnelles entre pixels adjacents,
    caractéristiques de la structure de l'image (une image naturelle a des
    corrélations MSCN voisines faibles mais non nulles ; flou/bruit les
    altèrent de façon mesurable).

    Parameters
    ----------
    mscn : npt.NDArray
        Coefficients MSCN (voir :func:`mscn_transform`).

    Returns
    -------
    Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]
        Produits horizontal (H), vertical (V), diagonal principal (D1) et
        anti-diagonal (D2), même shape que ``mscn``. Le voisin est pris de
        façon circulaire (``np.roll``) plutôt que recadré : significatif
        sur les petits blocs (96×96 ou moins) utilisés par NIQE, où
        recadrer biaiserait la statistique vers le centre du bloc.
    """
    h = np.roll(mscn, 1, axis=1) * mscn
    v = np.roll(mscn, 1, axis=0) * mscn
    d1 = np.roll(np.roll(mscn, 1, axis=0), 1, axis=1) * mscn
    d2 = np.roll(np.roll(mscn, 1, axis=0), -1, axis=1) * mscn
    return h, v, d1, d2


def estimate_ggd_shape(rho: float) -> float:
    """
    Estime le paramètre de forme γ d'une distribution gaussienne généralisée
    (GGD) par correspondance de moments : trouve, sur une grille
    précalculée, le γ dont ``r(γ) = Γ(2/γ)²/(Γ(1/γ)Γ(3/γ))`` est le plus
    proche de ``rho`` (ratio moment observé).

    Parameters
    ----------
    rho : float
        Ratio de moments observé (voir :func:`aggd_fit`).

    Returns
    -------
    float
        Paramètre de forme estimé, dans ``[0.2, 10.0)``.
    """
    idx = int(np.argmin((_R_TABLE - rho) ** 2))
    return float(_GAMMA_RANGE[idx])


def aggd_fit(coeffs: npt.NDArray) -> Tuple[float, float, float, float]:
    """
    Ajuste une distribution gaussienne généralisée asymétrique (AGGD) aux
    coefficients fournis, par la méthode des moments.

    Parameters
    ----------
    coeffs : npt.NDArray
        Coefficients à ajuster (MSCN ou produit de voisinage), tableau de
        forme quelconque.

    Returns
    -------
    alpha : float
        Paramètre de forme (γ).
    left_std : float
        Écart-type du côté négatif de la distribution.
    right_std : float
        Écart-type du côté positif.
    mean_param : float
        Paramètre de position (asymétrie gauche/droite), nul pour une
        distribution symétrique.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> alpha, ls, rs, mean_param = aggd_fit(rng.standard_normal(1000))
    """
    x = coeffs.ravel().astype(np.float64)
    x2 = x * x
    left, right = x2[x < 0], x2[x >= 0]

    left_std = float(np.sqrt(left.mean())) if left.size else 0.0
    right_std = float(np.sqrt(right.mean())) if right.size else 0.0
    gamma_hat = left_std / right_std if right_std != 0 else np.inf

    mean_x2 = x2.mean()
    r_hat = (np.mean(np.abs(x)) ** 2) / mean_x2 if mean_x2 != 0 else np.inf
    rho = r_hat * ((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2)

    alpha = estimate_ggd_shape(rho)
    gam1, gam2, gam3 = _gamma_fn(1.0 / alpha), _gamma_fn(2.0 / alpha), _gamma_fn(3.0 / alpha)
    aggd_ratio = np.sqrt(gam1) / np.sqrt(gam3)
    mean_param = float((right_std - left_std) * (gam2 / gam1) * aggd_ratio)

    return alpha, left_std, right_std, mean_param
