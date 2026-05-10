"""
Famille SSIM — Full-Reference.

Fonctions exposées : ``ssim``, ``msssim``, ``cw_ssim``.
"""
from typing import Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
from loguru import logger

from ..core.checker import initial_check
from ..core.image_utils import prepare_image

# PyTorch
try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

# skimage
try:
    from skimage.metrics import structural_similarity as _ssim_skimage
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False

# sewar
try:
    from sewar.full_ref import ssim as _ssim_sewar
    from sewar.full_ref import msssim as _msssim_sewar
    _SEWAR = True
except ImportError:
    _SEWAR = False

# image-similarity-measures
try:
    from image_similarity_measures.quality_metrics import ssim as _ssim_ism  # type: ignore
    _ISM = True
except ImportError:
    _ISM = False

# IQA_pytorch
try:
    from IQA_pytorch import SSIM as _IQA_ssim  # type: ignore
    from IQA_pytorch import MS_SSIM as _IQA_msssim  # type: ignore
    from IQA_pytorch import CW_SSIM as _IQA_cw_ssim  # type: ignore
    _IQA = True
except ImportError:
    _IQA = False


def ssim(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "skimage",
    **kwargs,
) -> Union[None, float, Tuple[float, npt.NDArray]]:
    """
    Calcule le Structural SIMilarity Index (SSIM).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence, shape ``(H, W)`` ou ``(H, W, 3)``.
    P : npt.NDArray
        Image traitée / dégradée, même shape que ``GT``.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"skimage"`` (défaut) — supporte niveaux de gris et RGB
        * ``"metrikz"``
        * ``"sewar"``
        * ``"image-similarity-measures"``
        * ``"ssim_1"`` — implémentation numpy (mubeta06)
        * ``"ssim_2"`` — implémentation améliorée
        * ``"ssim_3"`` — implémentation video-quality (aizvorski)
        * ``"ssim_4"`` — implémentation Clermont
        * ``"IQA_pytorch"``
    **kwargs
        Arguments optionnels transmis à l'algorithme sous-jacent
        (e.g. ``full=True`` pour ``skimage``, ``MAX`` pour ``sewar``).

    Returns
    -------
    float or None
        Valeur du SSIM dans ``[0, 1]``. Une valeur plus élevée indique une meilleure qualité.
        ``None`` si l'algorithme ne supporte pas le nombre de canaux de l'image.
    Tuple[float, npt.NDArray], optional
        ``(valeur, carte_ssim)`` si ``full=True`` est passé et que l'algo le supporte.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu, si les images sont incompatibles
        ou si le nombre de dimensions est supérieur à 3.
    """
    GT, P = initial_check(GT, P)

    if algo == "skimage":
        if not _SKIMAGE:
            raise ImportError("scikit-image est requis pour algo='skimage'.")
        pixel_diff = float(np.max(np.maximum(GT, P)) - np.min(np.minimum(GT, P)))
        full = kwargs.get("full", False)
        channel_axis = 2 if GT.ndim == 3 else None
        return _ssim_skimage(GT, P, data_range=pixel_diff, full=full, channel_axis=channel_axis)

    if algo == "metrikz":
        from ..backends.metrikz.metrikz import ssim as _ssim_metrikz
        return float(_ssim_metrikz(GT, P))

    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        MAX = kwargs.get("MAX", None)
        result = _ssim_sewar(GT, P) if MAX is None else _ssim_sewar(GT, P, MAX=MAX)
        return float(result[0])

    if algo == "image-similarity-measures":
        if not _ISM:
            return None
        return float(_ssim_ism(GT[..., np.newaxis], P[..., np.newaxis]))

    if algo in ("ssim_1", "ssim_2", "ssim_3", "ssim_4"):
        from ..backends.ssim import skimage_impl, improved_impl, video_impl, clermont_impl
        _map = {
            "ssim_1": skimage_impl.ssim,
            "ssim_2": improved_impl.compute_ssim_improved,
            "ssim_3": video_impl.ssim,
            "ssim_4": clermont_impl.compute_ssim,
        }
        fn = _map[algo]
        result = fn(GT, P)
        if isinstance(result, np.ndarray):
            return float(np.mean(result))
        return float(result)

    if algo == "IQA_pytorch":
        if not _IQA or not _TORCH:
            raise ImportError("IQA_pytorch et torch sont requis pour algo='IQA_pytorch'.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref = prepare_image(GT).to(device)
        dist = prepare_image(P).to(device)
        model = _IQA_ssim(channels=1 if GT.ndim == 2 else 3)
        return float(model(dist, ref, as_loss=False).item())

    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def msssim(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "sewar",
    **kwargs,
) -> Optional[float]:
    """
    Calcule le Multi-Scale SSIM (MS-SSIM).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"sewar"`` (défaut)
        * ``"msssim_1"`` — implémentation numpy (mubeta06)
        * ``"msssim_2"`` — implémentation alternative
        * ``"IQA_pytorch"``
    **kwargs
        Arguments optionnels (e.g. ``MAX`` pour ``sewar``).

    Returns
    -------
    float or None
        Valeur du MS-SSIM. Une valeur plus élevée indique une meilleure qualité.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)

    if algo == "sewar":
        if not _SEWAR:
            raise ImportError("sewar est requis pour algo='sewar'.")
        MAX = kwargs.get("MAX", None)
        val = _msssim_sewar(GT, P) if MAX is None else _msssim_sewar(GT, P, MAX=MAX)
        return float(np.real_if_close(val))

    if algo == "msssim_1":
        from ..backends.ssim.skimage_impl import msssim as _msssim_1
        return float(_msssim_1(GT, P))

    if algo == "msssim_2":
        from ..backends.ssim.mssim_impl import MSSIM
        return float(MSSIM().compute(GT, P))

    if algo == "IQA_pytorch":
        if not _IQA or not _TORCH:
            logger.error("IQA_pytorch et torch sont requis pour algo='IQA_pytorch'.")
            raise ImportError("IQA_pytorch et torch sont requis.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref = prepare_image(GT).to(device)
        dist = prepare_image(P).to(device)
        model = _IQA_msssim(channels=1 if GT.ndim == 2 else 3)
        return float(model(dist, ref, as_loss=False).item())

    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")


def cw_ssim(
    GT: npt.NDArray,
    P: npt.NDArray,
    algo: str = "cw_ssim_1",
    **kwargs,
) -> Union[float, Tuple[float, npt.NDArray], None]:
    """
    Calcule le Complex Wavelet SSIM (CW-SSIM).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence.
    P : npt.NDArray
        Image traitée / dégradée.
    algo : str, optional
        Algorithme à utiliser. Valeurs possibles :

        * ``"cw_ssim_1"`` (défaut) — implémentation numpy (charparr)
        * ``"IQA_pytorch"`` — non fonctionnel avec Torch >= 2.3.1
    **kwargs
        Arguments optionnels (e.g. ``width`` pour ``cw_ssim_1``).

    Returns
    -------
    float or None
        Valeur du CW-SSIM. Une valeur plus élevée indique une meilleure qualité.
    Tuple[float, npt.NDArray], optional
        ``(valeur, carte)`` si l'algo le supporte.

    Raises
    ------
    ValueError
        Si ``algo`` n'est pas reconnu ou si les images sont incompatibles.
    """
    GT, P = initial_check(GT, P)

    if algo == "cw_ssim_1":
        from ..backends.cw_ssim.wavelet_impl import cw_ssim as _cw_ssim_1
        width = kwargs.get("width", None)
        return _cw_ssim_1(GT, P) if width is None else _cw_ssim_1(GT, P, width=width)

    if algo == "IQA_pytorch":
        if not _IQA or not _TORCH:
            raise ImportError("IQA_pytorch et torch sont requis.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ref = prepare_image(GT).to(device)
        dist = prepare_image(P).to(device)
        model = _IQA_cw_ssim(channels=1 if GT.ndim == 2 else 3, device=device)
        return float(model(dist, ref, as_loss=False).item())

    logger.error(f"'algo' inconnu : {algo!r}")
    raise ValueError(f"'algo' inconnu : {algo!r}")

#EOF