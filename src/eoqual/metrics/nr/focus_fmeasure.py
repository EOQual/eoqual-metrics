"""
27 opérateurs de mesure de mise au point (focus) pour la métrique
``fmeasure``.

Implémentation propre, écrite à partir de la définition mathématique de
chaque opérateur telle que documentée dans l'article de synthèse cité en
Référence — remplace l'implémentation précédemment vendored, portée d'une
soumission MATLAB File Exchange dont la licence n'a pu être vérifiée (voir
`THIRD_PARTY_LICENSES.md`).

Chaque opérateur est cité individuellement ci-dessous avec sa référence
d'origine (littérature classique de l'autofocus/*shape-from-focus*, années
1980-2010) — l'implémentation ne recopie aucun code tiers, seulement la
définition mathématique publiée de chaque mesure.

Référence de synthèse
----------------------
Pertuz, S., Puig, D., & Garcia, M. A. (2013). *Analysis of focus measure
operators for shape-from-focus*. Pattern Recognition, 46(5), 1415-1432.
DOI: 10.1016/j.patcog.2012.11.011
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import numpy.typing as npt
import pywt
from scipy.fftpack import dctn
from scipy.ndimage import convolve, generic_filter

__all__ = ["MEASURES_FOCUS", "fmeasure"]

MEASURES_FOCUS = [
    "ACMO", "BREN", "CONT", "CURV", "DCTE",
    "DCTR", "GDER", "GLLV", "GLVA", "GLVN",
    "GRAE", "GRAS", "GRAT", "HELM",
    "HISR", "LAPD", "LAPE", "LAPM", "LAPV",
    "SFIL", "SFRQ", "TENG", "TENV", "VOLA",
    "WAVR", "WAVS", "WAVV",
]

_WSIZE = 15  # taille de fenêtre locale par défaut (opérateurs qui en ont besoin)

# Noyau Laplacien normalisé partagé par LAPE et LAPV.
_LAPLACIAN_KERNEL = np.array([
    [0.1667, 0.6667, 0.1667],
    [0.6667, -3.3333, 0.6667],
    [0.1667, 0.6667, 0.1667],
])

_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])


def _gaussian_derivative_kernels(window: int) -> tuple:
    """Filtres gaussien dérivé Gx/Gy, normalisés en norme L1, taille ``window``."""
    n = window // 2
    sigma = n / 2.5
    x, y = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1))
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2)) / (2 * np.pi * sigma)
    gx = -x * g / (sigma ** 2)
    gy = -y * g / (sigma ** 2)
    return gx, gy


def _acmo(image: npt.NDArray) -> float:
    """ACMO — Absolute Central MOment (Shirvaikar, 2004)."""
    m, n = image.shape
    hist, _ = np.histogram(image.ravel(), bins=256)
    hist = hist / (m * n)
    weighted = np.abs(np.arange(256) - np.mean(image)) * hist
    return float(np.sum(weighted))


def _bren(image: npt.NDArray) -> float:
    """BREN — mesure de Brenner (Santos et al., 1997)."""
    dh = np.zeros_like(image)
    dv = np.zeros_like(image)
    dv[:-2, :] = image[2:, :] - image[:-2, :]
    dh[:, :-2] = image[:, 2:] - image[:, :-2]
    return float(np.mean(np.maximum(dh, dv) ** 2))


def _cont(image: npt.NDArray) -> float:
    """CONT — contraste local (Nanda & Cutler, 2001)."""
    def _local_contrast(window: npt.NDArray) -> float:
        return float(np.sum(np.abs(window - window[4])))
    fm = generic_filter(image, _local_contrast, size=3)
    return float(np.mean(fm))


def _curv(image: npt.NDArray) -> float:
    """CURV — courbure de l'image (Helmli & Scherer, 2001)."""
    m1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    m2 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])
    p0 = convolve(image, m1, mode="nearest") / 6
    p1 = convolve(image, m1.T, mode="nearest") / 6
    p2 = 3 * convolve(image, m2, mode="nearest") / 10 - convolve(image, m2.T, mode="nearest") / 5
    p3 = -convolve(image, m2, mode="nearest") / 5 + 3 * convolve(image, m2, mode="nearest") / 10
    return float(np.mean(np.abs(p0) + np.abs(p1) + np.abs(p2) + np.abs(p3)))


def _dct_block_energy_ratio(block: npt.NDArray) -> float:
    mt = dctn(block, type=2, norm="ortho") ** 2
    return float((np.sum(mt) - mt[0, 0]) / mt[0, 0])


def _dcte(image: npt.NDArray) -> float:
    """DCTE — énergie DCT (Shen & Chen, 2006), blocs 8×8."""
    scores = [
        _dct_block_energy_ratio(image[i:i + 8, j:j + 8])
        for i in range(0, image.shape[0], 8) for j in range(0, image.shape[1], 8)
    ]
    return float(np.mean(scores))


def _dct_reduced_ratio(block: npt.NDArray) -> float:
    m = dctn(block, type=2, norm="ortho")
    return float((m[0, 1] ** 2 + m[0, 2] ** 2 + m[1, 0] ** 2 + m[1, 1] ** 2 + m[2, 0] ** 2) / (m[0, 0] ** 2))


def _dctr(image: npt.NDArray) -> float:
    """DCTR — ratio d'énergie DCT réduit (Lee et al., 2009), blocs 8×8."""
    scores = [
        _dct_reduced_ratio(image[i:i + 8, j:j + 8])
        for i in range(0, image.shape[0], 8) for j in range(0, image.shape[1], 8)
    ]
    return float(np.mean(scores))


def _gder(image: npt.NDArray) -> float:
    """GDER — dérivée gaussienne (Geusebroek et al., 2000)."""
    gx, gy = _gaussian_derivative_kernels(_WSIZE)
    gx = gx / np.sum(np.abs(gx))
    gy = gy / np.sum(np.abs(gy))
    rx = convolve(image, gx, mode="nearest")
    ry = convolve(image, gy, mode="nearest")
    return float(np.mean(rx ** 2 + ry ** 2))


def _gllv(image: npt.NDArray) -> float:
    """GLLV — variance locale des niveaux de gris (Pech-Pacheco et al., 2000)."""
    local_var = generic_filter(image, np.var, size=(_WSIZE, _WSIZE))
    return float(np.var(local_var))


def _glva(image: npt.NDArray) -> float:
    """GLVA — variance des niveaux de gris (Krotkov, 1986)."""
    return float(np.std(image))


def _glvn(image: npt.NDArray) -> float:
    """GLVN — variance normalisée des niveaux de gris (Santos et al., 1997)."""
    return float(np.std(image) ** 2 / np.mean(image))


def _grae(image: npt.NDArray) -> float:
    """GRAE — énergie du gradient (Subbarao et al., 1992)."""
    ix, iy = np.copy(image), np.copy(image)
    iy[:-1, :] = np.diff(image, axis=0)
    ix[:, :-1] = np.diff(image, axis=1)
    return float(np.mean(ix ** 2 + iy ** 2))


def _gras(image: npt.NDArray) -> float:
    """GRAS — gradient au carré, composante horizontale (Eskicioglu & Fisher, 1995)."""
    return float(np.mean(np.diff(image, axis=1) ** 2))


def _grat(image: npt.NDArray) -> float:
    """GRAT — gradient seuillé (Santos et al., 1997), seuil 0."""
    ix, iy = np.copy(image), np.copy(image)
    iy[:-1, :] = np.diff(image, axis=0)
    ix[:, :-1] = np.diff(image, axis=1)
    fm = np.maximum(np.abs(ix), np.abs(iy))
    nonzero = fm[fm != 0]
    return float(np.sum(nonzero) / nonzero.size) if nonzero.size else 0.0


def _helm(image: npt.NDArray) -> float:
    """HELM — mesure moyenne de Helmli & Scherer (2001)."""
    mean_filter = np.ones((_WSIZE, _WSIZE)) / (_WSIZE ** 2)
    u = convolve(image, mean_filter, mode="nearest")
    ratio = np.divide(u, image, out=np.ones_like(u), where=image != 0)
    fm = np.where(u > image, ratio, 1.0 / np.where(ratio != 0, ratio, 1.0))
    return float(np.mean(fm))


def _hisr(image: npt.NDArray) -> float:
    """HISR — étendue de l'histogramme (Firestone et al., 1991)."""
    return float(np.max(image) - np.min(image))


def _lapd(image: npt.NDArray) -> float:
    """LAPD — Laplacien diagonal (Thelen et al., 2009)."""
    m1 = np.array([-1, 2, -1])
    m2 = np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]]) / np.sqrt(2)
    m3 = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]]) / np.sqrt(2)
    f1 = convolve(image, m1[np.newaxis, :], mode="nearest")
    f2 = convolve(image, m2, mode="nearest")
    f3 = convolve(image, m3, mode="nearest")
    f4 = convolve(image, m1[np.newaxis, :].T, mode="nearest")
    return float(np.mean(np.abs(f1) + np.abs(f2) + np.abs(f3) + np.abs(f4)))


def _lape(image: npt.NDArray) -> float:
    """LAPE — énergie du Laplacien (Subbarao et al., 1992)."""
    fm = convolve(image, _LAPLACIAN_KERNEL, mode="nearest")
    return float(np.mean(fm ** 2))


def _lapm(image: npt.NDArray) -> float:
    """LAPM — Laplacien modifié (Nayar & Nakagawa, 1989)."""
    m = np.array([-1, 2, -1])
    lx = convolve(image, m[np.newaxis, :], mode="nearest")
    ly = convolve(image, m[np.newaxis, :].T, mode="nearest")
    return float(np.mean(np.abs(lx) + np.abs(ly)))


def _lapv(image: npt.NDArray) -> float:
    """LAPV — variance du Laplacien (Pech-Pacheco et al., 2000)."""
    fm = convolve(image, _LAPLACIAN_KERNEL, mode="nearest")
    return float(np.var(fm))


def _sfil(image: npt.NDArray) -> float:
    """SFIL — filtres orientables (Minhas et al., 2009)."""
    gx, gy = _gaussian_derivative_kernels(_WSIZE)
    gx, gy = gx / np.sum(gx), gy / np.sum(gy)
    rx = convolve(image, gx, mode="nearest")
    ry = convolve(image, gy, mode="nearest")
    responses = [rx, ry]
    for i in range(2, 8):
        theta = np.deg2rad(i * 45)
        responses.append(np.cos(theta) * rx + np.sin(theta) * ry)
    return float(np.mean(np.max(np.stack(responses, axis=-1), axis=-1)))


def _sfrq(image: npt.NDArray) -> float:
    """SFRQ — fréquence spatiale (Eskicioglu & Fisher, 1995)."""
    ix, iy = np.zeros_like(image), np.zeros_like(image)
    ix[:, :-1] = np.diff(image, axis=1)
    iy[:-1, :] = np.diff(image, axis=0)
    return float(np.mean(np.sqrt(ix ** 2 + iy ** 2)))


def _sobel_gradients(image: npt.NDArray) -> tuple:
    gx = convolve(image, _SOBEL_X, mode="nearest")
    gy = convolve(image, _SOBEL_X.T, mode="nearest")
    return gx, gy


def _teng(image: npt.NDArray) -> float:
    """TENG — Tenengrad (Krotkov, 1986)."""
    gx, gy = _sobel_gradients(image)
    return float(np.mean(gx ** 2 + gy ** 2))


def _tenv(image: npt.NDArray) -> float:
    """TENV — variance du Tenengrad (Pech-Pacheco et al., 2000)."""
    gx, gy = _sobel_gradients(image)
    return float(np.var(gx ** 2 + gy ** 2))


def _vola(image: npt.NDArray) -> float:
    """VOLA — corrélation de Vollath (Santos et al., 1997)."""
    i1, i2 = np.copy(image), np.copy(image)
    i1[:-1, :] = image[1:, :]
    i2[:-2, :] = image[2:, :]
    return float(np.mean(image * (i1 - i2)))


def _wavelet_detail_coeffs(image: npt.NDArray, level: int) -> tuple:
    coeffs = pywt.wavedec2(image, "db6", level=level)
    return coeffs[1]  # (cH, cV, cD) du niveau le plus fin


def _wavs(image: npt.NDArray) -> float:
    """WAVS — somme des coefficients d'ondelettes (Yang & Nelson, 2003)."""
    ch, cv, cd = _wavelet_detail_coeffs(image, level=1)
    return float(np.mean(np.abs(ch) + np.abs(cv) + np.abs(cd)))


def _wavr(image: npt.NDArray) -> float:
    """WAVR — ratio de coefficients d'ondelettes (Yang & Nelson, 2003)."""
    coeffs = pywt.wavedec2(image, "db6", level=3)
    ch, cv, cd = coeffs[1]
    wh = float(np.mean(np.abs(ch) ** 2 + np.abs(cv) ** 2 + np.abs(cd) ** 2))
    wl = sum(float(np.mean(np.abs(c))) for c in coeffs[0:3])
    return wh / wl


def _wavv(image: npt.NDArray) -> float:
    """WAVV — variance des coefficients d'ondelettes (Yang & Nelson, 2003)."""
    ch, cv, cd = _wavelet_detail_coeffs(image, level=1)
    return float(np.std(ch) ** 2 + np.std(cv) ** 2 + np.std(cd) ** 2)


_OPERATORS: Dict[str, Callable[[npt.NDArray], float]] = {
    "ACMO": _acmo, "BREN": _bren, "CONT": _cont, "CURV": _curv,
    "DCTE": _dcte, "DCTR": _dctr, "GDER": _gder, "GLLV": _gllv,
    "GLVA": _glva, "GLVN": _glvn, "GRAE": _grae, "GRAS": _gras,
    "GRAT": _grat, "HELM": _helm, "HISR": _hisr, "LAPD": _lapd,
    "LAPE": _lape, "LAPM": _lapm, "LAPV": _lapv, "SFIL": _sfil,
    "SFRQ": _sfrq, "TENG": _teng, "TENV": _tenv, "VOLA": _vola,
    "WAVR": _wavr, "WAVS": _wavs, "WAVV": _wavv,
}


def fmeasure(Image: npt.NDArray, algo: str, ROI: Optional[List[int]] = None) -> float:
    """
    Mesure le degré relatif de mise au point (focus) d'une image.

    27 opérateurs disponibles (voir :data:`MEASURES_FOCUS`), chacun issu
    de la littérature classique de l'autofocus / *shape-from-focus* — voir
    la docstring de chaque fonction privée (``_acmo``, ``_bren``...) pour
    sa référence précise, et Pertuz et al. 2013 pour la synthèse.

    Parameters
    ----------
    Image : npt.NDArray
        Image en niveaux de gris, shape ``(H, W)``.
    algo : str
        Nom de l'opérateur (voir :data:`MEASURES_FOCUS`), insensible à la casse.
    ROI : list[int] or None, optional
        Rectangle ``[x0, y0, largeur, hauteur]`` restreignant le calcul à
        une région de l'image. Par défaut ``None`` (image entière).

    Returns
    -------
    float
        Degré relatif de mise au point. L'échelle et le sens
        d'optimalité dépendent de l'opérateur (voir ``config.py``).

    Raises
    ------
    ValueError
        Si ``Image`` n'est pas 2-D ou si ``algo`` est inconnu.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256) * 255
    >>> score = fmeasure(img, "GLVA")
    """
    if Image.ndim > 2:
        raise ValueError("L'image doit être en niveaux de gris !")

    if ROI is not None and len(ROI) > 0:
        Image = Image[ROI[1]:ROI[1] + ROI[3], ROI[0]:ROI[0] + ROI[2]]

    Image = Image.astype(np.float64)
    algo = algo.upper()

    operator = _OPERATORS.get(algo)
    if operator is None:
        raise ValueError(f"Opérateur de focus inconnu : {algo!r}")
    return operator(Image)
