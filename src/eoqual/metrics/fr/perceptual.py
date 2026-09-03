"""
Métriques perceptuelles — Full-Reference.

Fonctions exposées : ``dists``, ``fsim``, ``gmsd``, ``issm``, ``lpips_vgg``,
``mad``, ``nlpd``, ``pamse``, ``reco``, ``vif``, ``vsi``.
"""
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check
from ..core.image_utils import prepare_image

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

try:
    from IQA_pytorch import DISTS as _IQA_DISTS  # type: ignore
    from IQA_pytorch import FSIM as _IQA_FSIM  # type: ignore
    from IQA_pytorch import GMSD as _IQA_GMSD  # type: ignore
    from IQA_pytorch import LPIPSvgg as _IQA_LPIPS  # type: ignore
    from IQA_pytorch import MAD as _IQA_MAD  # type: ignore
    from IQA_pytorch import NLPD as _IQA_NLPD  # type: ignore
    from IQA_pytorch import VIF as _IQA_VIF  # type: ignore
    from IQA_pytorch import VSI as _IQA_VSI  # type: ignore
    _IQA = True
except ImportError:
    _IQA = False

try:
    from image_similarity_measures.quality_metrics import (  # type: ignore
        fsim as _fsim_ism,
        issm as _issm_ism,
    )
    _ISM = True
except ImportError:
    _ISM = False

try:
    from sewar.full_ref import vifp as _vifp_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False

try:
    import piq  # noqa: F401
    _PIQ = _TORCH
except ImportError:
    _PIQ = False


def _iqa_score(model_cls, GT: npt.NDArray, P: npt.NDArray, **model_kwargs) -> float:
    """Helper interne : instancie un modèle IQA_pytorch et retourne le score."""
    if not (_IQA and _TORCH):
        raise ImportError(
            "IQA_pytorch et PyTorch sont requis : pip install \"eoqual-metrics[deep]\"."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = prepare_image(GT).to(device)
    dist = prepare_image(P).to(device)
    channels = 1 if GT.ndim == 2 else 3
    model = model_cls(channels=channels, **model_kwargs).to(device)
    return float(model(dist, ref, as_loss=False).item())


def _require_piq() -> None:
    if not _PIQ:
        raise ImportError(
            "piq et PyTorch sont requis pour algo='piq' : pip install \"eoqual-metrics[deep]\"."
        )


def _require_ism() -> None:
    if not _ISM:
        raise ImportError(
            "image-similarity-measures est requis pour cet algo : "
            "pip install image-similarity-measures --no-deps (voir README.md)."
        )


def dists(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "piq",
) -> Optional[float]:
    """
    Calcule le Deep Image Structure and Texture Similarity (DISTS).

    Référence : https://github.com/dingkeyan93/DISTS

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence, niveaux de gris ou RGB.
    P : npt.NDArray
        Image traitée / dégradée de même shape.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"piq"`` (défaut) — backbone VGG16, télécharge les poids
          pré-entraînés au premier appel (voir ``backends/piq/common.py``)
        * ``"IQA_pytorch"`` — legacy, voir ``THIRD_PARTY_LICENSES.md``

    Returns
    -------
    float or None
        Valeur du DISTS. Une valeur plus faible indique une meilleure qualité.
        ``None`` si l'image n'est pas RGB (algo ``IQA_pytorch`` uniquement).

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si la dépendance requise (``piq``/``IQA_pytorch`` + PyTorch)
        n'est pas installée.
    """
    GT, P = initial_check(GT, P)
    if algo == "piq":
        _require_piq()
        from ..backends.piq.common import get_dists_model, to_rgb_tensors
        x, y = to_rgb_tensors(GT, P)
        return float(get_dists_model()(x, y).item())
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_DISTS, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def fsim(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "image-similarity-measures",
) -> float:
    """
    Calcule la Feature Similarity (FSIM).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"image-similarity-measures"`` (défaut) — pas de dépendance torch
        * ``"piq"`` — voir ``backends/piq/common.py`` (Apache-2.0)
        * ``"IQA_pytorch"`` — déprécié (erreur avec tenseurs complexes)

    Returns
    -------
    float
        Valeur de la FSIM. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si la dépendance requise par l'algo choisi n'est pas installée.
    """
    GT, P = initial_check(GT, P)
    if algo == "image-similarity-measures":
        _require_ism()
        return float(_fsim_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    if algo == "piq":
        _require_piq()
        from ..backends.piq.common import to_rgb_tensors
        x, y = to_rgb_tensors(GT, P)
        return float(piq.fsim(x, y).item())
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_FSIM, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def gmsd(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
    return_map: bool = True,
) -> Union[None, float, Tuple[float, npt.NDArray]]:
    """
    Calcule le Gradient Magnitude Similarity Deviation (GMSD).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence en niveaux de gris ``(H, W)`` pour ``algo="numpy"``,
        ou RGB ``(H, W, 3)`` pour ``algo="IQA_pytorch"``.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut) — niveaux de gris uniquement
        * ``"piq"`` — Apache-2.0, voir ``backends/piq/common.py``
        * ``"IQA_pytorch"`` — legacy
    return_map : bool, optional
        Si ``True`` et ``algo="numpy"``, retourne la carte GMSD. Par défaut ``True``.

    Returns
    -------
    float or None
        Valeur du GMSD. Une valeur plus faible indique une meilleure qualité.
        ``None`` si l'algo ne supporte pas le nombre de canaux de l'image.
    Tuple[float, npt.NDArray], optional
        ``(valeur, carte_gmsd)`` si ``return_map=True`` et ``algo="numpy"``.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    if GT.ndim > 3:
        raise ValueError(f"L'image doit être 2D ou 3D, reçu shape={GT.shape}.")
    if GT.ndim == 3 and algo == "numpy":
        logger.warning("gmsd/numpy — nécessite une image monobande.")
        return None

    GT, P = initial_check(GT, P)

    if algo == "numpy":
        from ..backends.gmsd.numpy_impl import gmsd as _gmsd_numpy
        return _gmsd_numpy(GT, P, returnMap=return_map)
    if algo == "piq":
        _require_piq()
        from ..backends.piq.common import to_rgb_tensors
        x, y = to_rgb_tensors(GT, P)
        return float(piq.gmsd(x, y).item())
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_GMSD, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def issm(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "image-similarity-measures",
) -> float:
    """
    Calcule l'Information theoretic-based Statistic Similarity Measure (ISSM).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"image-similarity-measures"`` (défaut)

    Returns
    -------
    float
        Valeur de l'ISSM. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si ``image-similarity-measures`` n'est pas installé.
    """
    GT, P = initial_check(GT, P)
    if algo == "image-similarity-measures":
        _require_ism()
        return float(_issm_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def lpips_vgg(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "piq",
) -> Optional[float]:
    """
    Calcule le Learned Perceptual Image Patch Similarity (LPIPS-VGG).

    Référence : https://github.com/richzhang/PerceptualSimilarity

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence, niveaux de gris ou RGB.
    P : npt.NDArray
        Image traitée / dégradée, même shape.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"piq"`` (défaut) — backbone VGG16, voir
          ``backends/piq/common.py`` (téléchargement des poids au premier
          appel)
        * ``"IQA_pytorch"`` — legacy, **inutilisable avec torch ≥ 1.8**
          (voir ``THIRD_PARTY_LICENSES.md``) ; conservé pour mémoire, pas
          pour un usage réel

    Returns
    -------
    float or None
        Valeur du LPIPS. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si la dépendance requise (``piq``/``IQA_pytorch`` + PyTorch)
        n'est pas installée.
    """
    GT, P = initial_check(GT, P)
    if algo == "piq":
        _require_piq()
        from ..backends.piq.common import get_lpips_model, to_rgb_tensors
        x, y = to_rgb_tensors(GT, P)
        return float(get_lpips_model()(x, y).item())
    if algo == "IQA_pytorch":
        if not (_IQA and _TORCH):
            raise ImportError(
                "IQA_pytorch et PyTorch sont requis : pip install \"eoqual-metrics[deep]\"."
            )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref = prepare_image(GT).to(device)
        dist = prepare_image(P).to(device)
        model = _IQA_LPIPS(channels=3)
        return float(model(dist, ref, as_loss=False).item())
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def mad(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "IQA_pytorch",
) -> float:
    """
    Calcule le Most Apparent Distortion (MAD).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"IQA_pytorch"`` (défaut) — **inutilisable en pratique**, casse
          avec torch ≥ 1.8 (voir ``config.py``, ``'algo': []`` dans le
          registre). ``piq`` n'implémente pas MAD : aucune alternative
          permissive identifiée à ce jour (voir
          ``THIRD_PARTY_LICENSES.md``).

    Returns
    -------
    float
        Valeur du MAD. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si IQA_pytorch ou PyTorch ne sont pas installés.
    """
    GT, P = initial_check(GT, P)
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_MAD, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def nlpd(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "IQA_pytorch",
) -> float:
    """
    Calcule la Normalized Laplacian Pyramid Distance (NLPD).

    Référence : https://www.cns.nyu.edu/~lcv/NLPyr/

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"IQA_pytorch"`` (défaut) — legacy, ``piq`` n'implémente pas
          NLPD : aucune alternative permissive identifiée à ce jour (voir
          ``THIRD_PARTY_LICENSES.md``). Fonctionnel (contrairement à
          ``mad``/``lpips_vgg``/``vsi``) tant que torch reste dans la
          plage bornée par l'extra ``deep`` (``pyproject.toml``).

    Returns
    -------
    float
        Valeur du NLPD. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si IQA_pytorch ou PyTorch ne sont pas installés.
    """
    GT, P = initial_check(GT, P)
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_NLPD, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def pamse(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> float:
    """
    Calcule le Perceptual-fidelity Aware Mean Squared Error (PAMSE).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)

    Returns
    -------
    float
        Valeur du PAMSE. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "numpy":
        from ..backends.pamse.numpy_impl import pamse as _pamse_numpy
        return float(_pamse_numpy(GT, P))
    raise ValueError(f"'algo' inconnu : {algo!r}")


def reco(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "numpy",
) -> Optional[float]:
    """
    Calcule le Relative Polar Edge Coherence (RECO).

    Cohérence de phase entre deux harmoniques circulaires de Laguerre-Gauss
    (ordres 1 et 3) — Baroncini et al. 2009. Voir
    ``backends/reco/numpy_impl.py`` pour la méthodologie détaillée.

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"numpy"`` (défaut)

    Returns
    -------
    float, none
        Valeur du RECO. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu.
    """
    GT, P = initial_check(GT, P)

    if algo == "numpy":
        from ..backends.reco.numpy_impl import reco as _reco
        return _reco(GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def vif(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
) -> Optional[float]:
    """
    Calcule la Visual Information Fidelity (VIF).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"sewar"`` (défaut)
        * ``"metrikz"``
        * ``"mscale_impl"``
        * ``"wavelet_impl"``
        * ``"piq"`` — Apache-2.0, voir ``backends/piq/common.py``
        * ``"IQA_pytorch"`` — legacy

    Returns
    -------
    float or None
        Valeur de la VIF. Une valeur plus élevée indique une meilleure qualité.
        ``None`` si l'algo ne supporte pas le nombre de canaux de l'image.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)

    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        return float(_vifp_sewar(GT, P))
    if algo == "metrikz":
        from ..backends.metrikz.metrikz import pbvif as _pbvif_metrikz
        return float(_pbvif_metrikz(GT, P))
    if algo == "mscale_impl":
        from ..backends.vif.mscale_impl import vifp_mscale as _vif_mscale
        return float(_vif_mscale(GT, P))
    if algo == "wavelet_impl":
        from ..backends.vif.wavelet_impl import vif as _vif_wavelet
        return float(_vif_wavelet(GT, P, wavelet="steerable"))
    if algo == "piq":
        _require_piq()
        from ..backends.piq.common import to_rgb_tensors
        x, y = to_rgb_tensors(GT, P)
        return float(piq.vif_p(x, y).item())
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_VIF, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def vsi(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "piq",
) -> float:
    """
    Calcule le Visual Saliency-Induced Index (VSI).

    Référence : https://ieeexplore.ieee.org/document/6873260

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"piq"`` (défaut) — Apache-2.0, voir ``backends/piq/common.py``
        * ``"IQA_pytorch"`` — legacy, **inutilisable avec torch ≥ 1.8**
          (voir ``THIRD_PARTY_LICENSES.md``) ; conservé pour mémoire

    Returns
    -------
    float
        Valeur du VSI.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si la dépendance requise (``piq``/``IQA_pytorch`` + PyTorch)
        n'est pas installée.
    """
    GT, P = initial_check(GT, P)
    if algo == "piq":
        _require_piq()
        from ..backends.piq.common import to_rgb_tensors
        x, y = to_rgb_tensors(GT, P)
        return float(piq.vsi(x, y).item())
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_VSI, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")

#EOF