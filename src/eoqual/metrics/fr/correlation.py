"""
Métriques de corrélation et statistiques — Full-Reference.

Fonctions exposées : ``ncc``, ``ndp``, ``nmi``, ``scc``, ``uqi``, ``jsd``.
"""
from typing import Optional, Union

import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check, versiontuple

try:
    from scipy.spatial import distance as _scipy_distance
    from scipy.spatial.distance import cdist as _cdist
    _SCIPY = True
except ImportError:
    _SCIPY = False

try:
    from sklearn.metrics.pairwise import cosine_similarity as _cosine_similarity
    _SKLEARN = True
except ImportError:
    _SKLEARN = False

try:
    from sewar.full_ref import scc as _scc_sewar
    from sewar.full_ref import uqi as _uqi_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False

try:
    from image_similarity_measures.quality_metrics import uiq as _uqi_ism  # type: ignore
    _ISM = True
except ImportError:
    _ISM = False


def ncc(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule le Normalized Cross-Correlation (NCC) — aussi appelé coefficient de Pearson.

    Le résultat est dans ``[-1, 1]``, où ``1`` signifie une corrélation parfaite.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)
        * ``"scipy"``

    Returns
    -------
    float
        Valeur du NCC dans ``[-1, 1]``. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    GT = GT.astype(np.float64)
    P = P.astype(np.float64)

    if algo == "numpy":
        numerator = np.mean((GT - np.mean(GT)) * (P - np.mean(P)))
        denom = np.std(GT) * np.std(P)
        return 0.0 if denom == 0 else float(numerator / denom)
    if algo == "scipy":
        if not _SCIPY:
            raise ImportError("scipy est requis pour algo='scipy'.")
        return float(1.0 - _scipy_distance.correlation(GT.flatten(), P.flatten()))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def ndp(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule le Normalized Dot Product (NDP) — aussi appelé angle cosinusoïdal (VCAD).

    Le résultat est dans ``[-1, 1]`` où ``1`` = vecteurs colinéaires positifs,
    ``0`` = orthogonaux, ``-1`` = opposés.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)
        * ``"scipy_1"``
        * ``"scipy_2"``
        * ``"sklearn"``

    Returns
    -------
    float
        Valeur du NDP dans ``[-1, 1]``. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    GT = GT.astype(np.float64)
    P = P.astype(np.float64)

    if algo == "numpy":
        return float(
            np.dot(GT.flatten(), P.flatten())
            / (np.linalg.norm(GT) * np.linalg.norm(P))
        )
    if algo == "scipy_1":
        if not _SCIPY:
            raise ImportError("scipy est requis pour algo='scipy_1'.")
        return float(1.0 - _scipy_distance.cosine(GT.flatten(), P.flatten()))
    if algo == "scipy_2":
        if not _SCIPY:
            raise ImportError("scipy est requis pour algo='scipy_2'.")
        return float(
            1.0 - _cdist(GT.reshape(1, -1), P.reshape(1, -1), "cosine")[0, 0]
        )
    if algo == "sklearn":
        if not _SKLEARN:
            raise ImportError("scikit-learn est requis pour algo='sklearn'.")
        return float(_cosine_similarity(GT.reshape(1, -1), P.reshape(1, -1))[0, 0])
    raise ValueError(f"'algo' inconnu : {algo!r}")


def nmi(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "skimage",
) -> float:
    """
    Calcule l'Information Mutuelle Normalisée (NMI).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"skimage"`` (défaut, nécessite scikit-image >= 0.19)

    Returns
    -------
    float
        Valeur de la NMI. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si scikit-image < 0.19 est utilisé.
    """
    GT, P = initial_check(GT, P)
    if algo == "skimage":
        import skimage  # type: ignore
        if versiontuple(skimage.__version__) < versiontuple("0.19"):
            raise ImportError("scikit-image >= 0.19 est requis pour nmi.")
        from skimage.metrics import normalized_mutual_information as _nmi_ski
        return float(_nmi_ski(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def scc(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
) -> float:
    """
    Calcule le Spatial Correlation Coefficient (SCC).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"sewar"`` (défaut)

    Returns
    -------
    float
        Valeur du SCC. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si sewar n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_scc_sewar(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def uqi(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "metrikz",
) -> Optional[float]:
    """
    Calcule l'Universal Image Quality Index (UQI).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"metrikz"`` (défaut)
        * ``"sewar"``
        * ``"image-similarity-measures"``

    Returns
    -------
    float or None
        Valeur de l'UQI, ou ``None`` si la librairie requise est indisponible.
        Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
#    _ALGO_GRAY_ONLY = {
#        "metrikz", "sewar", "image-similarity-measures"
#    }
#    _ALGO_RGB_CAPABLE = {
#        "metrikz", "sewar"
#    }
#
#    if GT.ndim > 3:
#        logger.error(f"L'image doit être 2D ou 3D, reçu shape={GT.shape}.")
#        raise ValueError(f"L'image doit être 2D ou 3D, reçu shape={GT.shape}.")
#    if GT.ndim == 3 and algo in _ALGO_GRAY_ONLY - _ALGO_RGB_CAPABLE:
#        logger.warning(f"uqi/{algo} — nécessite une image monobande.")
#        return None
#    if GT.ndim == 2 and algo in _ALGO_RGB_CAPABLE - _ALGO_GRAY_ONLY:
#        logger.warning(f"uqi/{algo} — nécessite une image RGB.")
#        return None

    GT, P = initial_check(GT, P)
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import uqi as _uqi_metrikz
        return float(_uqi_metrikz(GT, P))
    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_uqi_sewar(GT, P))
    if algo == "image-similarity-measures":
        if not _ISM:
            return None
        return float(_uqi_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def jsd(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
    bins: Optional[int] = None,
) -> float:
    """
    Calcule la distance de Jensen-Shannon (JSD) entre les histogrammes d'intensité.

    Version symétrisée et bornée de la divergence de Kullback-Leibler :
    ``JSD(p, q) = sqrt(0.5 * KL(p‖m) + 0.5 * KL(q‖m))``, avec ``m = 0.5*(p+q)``
    la distribution "moyenne". Contrairement à KL, ``JSD(p, q) == JSD(q, p)``
    et ne nécessite aucun epsilon de secours : ``m`` ne s'annule jamais là où
    ``p`` ou ``q`` est non nul. La racine carrée retournée ici est une
    distance au sens mathématique (inégalité triangulaire, voir Lin (1991),
    "Divergence measures based on the Shannon entropy", *IEEE Trans. Inf.
    Theory*), convention identique à ``scipy.spatial.distance.jensenshannon``.

    Compare les distributions globales d'intensité, indépendamment de toute
    correspondance spatiale pixel à pixel — utile pour repérer un décalage
    radiométrique global (exposition, calibration capteur, conditions
    atmosphériques) entre deux acquisitions, y compris mal recalées.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)
    bins : int, optional
        Nombre de classes de l'histogramme. Si ``None`` (défaut), calculé
        automatiquement par la règle de Freedman-Diaconis sur les valeurs
        combinées des deux images (bornée à ``[16, 256]``) — un nombre de
        classes fixe est bruité sur une petite image et grossier sur une
        grande ; l'adaptation évite les deux écueils.

    Returns
    -------
    float
        Distance de Jensen-Shannon (base 2) dans ``[0, 1]``. ``0`` =
        distributions d'intensité identiques. Une valeur plus faible
        indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.randint(0, 256, (64, 64)).astype(np.uint8)
    >>> jsd(img, img)
    0.0
    """
    GT, P = initial_check(GT, P)
    if algo != "numpy":
        raise ValueError(f"'algo' inconnu : {algo!r}")

    gt_flat = GT.astype(np.float64).ravel()
    p_flat = P.astype(np.float64).ravel()

    lo = float(min(gt_flat.min(), p_flat.min()))
    hi = float(max(gt_flat.max(), p_flat.max()))
    if hi <= lo:
        return 0.0  # les deux images sont constantes et de même valeur

    n_bins = bins if bins is not None else _freedman_diaconis_bins(gt_flat, p_flat)
    edges = np.linspace(lo, hi, n_bins + 1)
    hist_gt, _ = np.histogram(gt_flat, bins=edges)
    hist_p, _ = np.histogram(p_flat, bins=edges)

    p_dist = hist_gt / hist_gt.sum()
    q_dist = hist_p / hist_p.sum()
    m_dist = 0.5 * (p_dist + q_dist)

    jsd_value = 0.5 * _kl_div(p_dist, m_dist) + 0.5 * _kl_div(q_dist, m_dist)
    return float(np.sqrt(max(jsd_value, 0.0)))  # max(...,0) : garde-fou aux erreurs d'arrondi


def _kl_div(p: npt.NDArray, q: npt.NDArray) -> float:
    """
    Divergence de Kullback-Leibler ``KL(p‖q)`` en base 2.

    Les classes où ``p == 0`` sont ignorées (convention ``0 * log(0/q) = 0``) ;
    aucun epsilon n'est nécessaire pour ``q`` tant que ``q`` est la
    distribution "moyenne" ``m`` utilisée par :func:`jsd` (jamais nulle là
    où ``p`` est non nul).
    """
    mask = p > 0
    return float(np.sum(p[mask] * np.log2(p[mask] / q[mask])))


def _freedman_diaconis_bins(
    a: npt.NDArray,
    b: npt.NDArray,
    min_bins: int = 16,
    max_bins: int = 256,
) -> int:
    """
    Nombre de classes d'histogramme par la règle de Freedman-Diaconis.

    Utilise l'IQR (écart interquartile, robuste aux valeurs aberrantes —
    contrairement à l'écart-type ou au min/max) des deux échantillons
    combinés, pour que les deux histogrammes de :func:`jsd` partagent les
    mêmes classes. Résultat borné à ``[min_bins, max_bins]`` pour éviter un
    nombre de classes dégénéré (image quasi uniforme) ou excessif (grande
    image, faisant chuter le poids statistique de chaque classe).
    """
    combined = np.concatenate([a, b])
    q75, q25 = np.percentile(combined, [75, 25])
    iqr = q75 - q25
    bin_width = 2.0 * iqr / (combined.size ** (1.0 / 3.0))
    data_range = combined.max() - combined.min()
    if iqr <= 0 or bin_width <= 0:
        return max_bins
    return int(np.clip(np.ceil(data_range / bin_width), min_bins, max_bins))
