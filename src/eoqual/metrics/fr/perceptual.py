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


def _iqa_score(model_cls, GT: npt.NDArray, P: npt.NDArray, **model_kwargs) -> float:
    """Helper interne : instancie un modèle IQA_pytorch et retourne le score."""
    if not (_IQA and _TORCH):
        raise ImportError("IQA_pytorch et PyTorch sont requis.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ref = prepare_image(GT).to(device)
    dist = prepare_image(P).to(device)
    channels = 1 if GT.ndim == 2 else 3
    model = model_cls(channels=channels, **model_kwargs).to(device)
    return float(model(dist, ref, as_loss=False).item())


def dists(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "IQA_pytorch",
) -> Optional[float]:
    """
    Calcule le Deep Image Structure and Texture Similarity (DISTS).

    Référence : https://github.com/dingkeyan93/DISTS

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence RGB de shape ``(H, W, 3)``.
    P : npt.NDArray
        Image traitée / dégradée de même shape.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"IQA_pytorch"`` (défaut)

    Returns
    -------
    float or None
        Valeur du DISTS. Une valeur plus faible indique une meilleure qualité.
        ``None`` si l'image n'est pas RGB.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si IQA_pytorch ou PyTorch ne sont pas installés.
    """
    GT, P = initial_check(GT, P)
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_DISTS, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def fsim(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "image-similarity-measures",
) -> Optional[float]:
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

        * ``"image-similarity-measures"`` (défaut)
        * ``"IQA_pytorch"`` — déprécié (erreur avec tenseurs complexes)

    Returns
    -------
    float or None
        Valeur de la FSIM. Une valeur plus élevée indique une meilleure qualité.
        ``None`` si la librairie est indisponible.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "image-similarity-measures":
        if not _ISM:
            return None
        return float(_fsim_ism(GT[..., np.newaxis], P[..., np.newaxis]))
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
        * ``"IQA_pytorch"``
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
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_GMSD, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def issm(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "image-similarity-measures",
) -> Optional[float]:
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
    float or None
        Valeur de l'ISSM. Une valeur plus élevée indique une meilleure qualité.
        ``None`` si la librairie est indisponible.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)
    if algo == "image-similarity-measures":
        if not _ISM:
            return None
        return float(_issm_ism(GT[..., np.newaxis], P[..., np.newaxis]))
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def lpips_vgg(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "IQA_pytorch",
) -> Optional[float]:
    """
    Calcule le Learned Perceptual Image Patch Similarity (LPIPS-VGG).

    Référence : https://github.com/richzhang/PerceptualSimilarity

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence RGB.
    P : npt.NDArray
        Image traitée / dégradée RGB.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"IQA_pytorch"`` (défaut)

    Returns
    -------
    float or None
        Valeur du LPIPS. Une valeur plus faible indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si IQA_pytorch ou PyTorch ne sont pas installés.
    """
    GT, P = initial_check(GT, P)
    if algo == "IQA_pytorch":
        if not (_IQA and _TORCH):
            raise ImportError("IQA_pytorch et PyTorch sont requis.")
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

        * ``"IQA_pytorch"`` (défaut)

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

        * ``"IQA_pytorch"`` (défaut)

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
    algo: str = "polar_impl",
) -> Optional[float]:
    """
    Calcule le Relative Polar Edge Coherence (RECO).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"polar_impl"`` (défaut)

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

    if algo == "polar_impl":
        from ..backends.reco.polar_impl import reco as _reco
        return float(_reco(GT, P))
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
        * ``"IQA_pytorch"``

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
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_VIF, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def vsi(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "IQA_pytorch",
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

        * ``"IQA_pytorch"`` (défaut)

    Returns
    -------
    float
        Valeur du VSI.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    ImportError
        Si IQA_pytorch ou PyTorch ne sont pas installés.
    """
    GT, P = initial_check(GT, P)
    if algo == "IQA_pytorch":
        return _iqa_score(_IQA_VSI, GT, P)
    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")

#EOF