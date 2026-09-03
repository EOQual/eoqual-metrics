"""
Utilitaires de transformation et de préparation des images.

Ce module centralise les conversions nécessaires avant de passer
une image à un backend (notamment PyTorch).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def prepare_image(image: npt.NDArray) -> "torch.Tensor":  # type: ignore[name-defined]
    """
    Convertit un tableau NumPy en tenseur PyTorch normalisé pour IQA_pytorch.

    L'image doit être en niveaux de gris (2D) ou RGB (3D, HxWx3).
    Elle est normalisée dans ``[0, 1]`` si ce n'est pas déjà le cas,
    puis mise au format ``(1, C, H, W)`` attendu par IQA_pytorch.

    Parameters
    ----------
    image : npt.NDArray
        Tableau NumPy de shape ``(H, W)`` ou ``(H, W, 3)``.

    Returns
    -------
    torch.Tensor
        Tenseur de shape ``(1, C, H, W)`` sur CPU, dtype ``float32``.

    Raises
    ------
    ImportError
        Si PyTorch n'est pas installé.
    ValueError
        Si l'image n'est pas 2D ou 3D.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "PyTorch est requis pour préparer les images : pip install "
            "\"eoqual-metrics[deep]\"."
        ) from exc

    if image.ndim not in (2, 3):
        raise ValueError(
            f"L'image doit être 2D (H,W) ou 3D (H,W,C), reçu shape={image.shape}"
        )

    img = image.astype(np.float32)

    # Normalisation dans [0, 1] si les valeurs dépassent 1
    if img.max() > 1.0:
        img = img / 255.0

    # Passage en (C, H, W)
    if img.ndim == 2:
        img = img[np.newaxis, :, :]          # (1, H, W)
    else:
        img = img.transpose(2, 0, 1)         # (3, H, W)

    return torch.from_numpy(img).unsqueeze(0)  # (1, C, H, W)


def normalize_histogram(image: npt.NDArray) -> npt.NDArray:
    """
    Normalise l'histogramme d'une image sur l'intervalle ``[0, 1]``.

    Rend les métriques radiométriquement comparables entre images dont les
    gammes d'intensité diffèrent (ex. ``uint8`` vs ``uint16``, ou deux
    capteurs avec des dynamiques différentes). Utilisé par les métriques de
    netteté satellite (``antonel``, ``blur_kernel``, ``wavelet``, ``aem``,
    ``sasbem``) comme prétraitement systématique.

    Parameters
    ----------
    image : npt.NDArray
        Image 2‑D ou 3‑D, n'importe quel dtype numérique
        (``uint8``, ``uint16``, ``int16``, ``float32``...).

    Returns
    -------
    npt.NDArray
        Image ``float32`` dont les valeurs sont comprises entre 0 et 1.
        Retournée inchangée (castée) si l'image est quasi constante
        (``max - min < 1e-12``), pour éviter une division par zéro.

    Examples
    --------
    >>> import numpy as np
    >>> img = (np.random.rand(64, 64) * 4095).astype(np.uint16)
    >>> norm = normalize_histogram(img)
    >>> float(norm.min()), float(norm.max())
    (0.0, 1.0)
    """
    img = image.astype(np.float32)
    img_min, img_max = float(img.min()), float(img.max())
    if img_max - img_min < 1e-12:
        return img
    return (img - img_min) / (img_max - img_min)


def to_grayscale(image: npt.NDArray) -> npt.NDArray:
    """
    Convertit une image RGB en niveaux de gris.

    Utilise la formule de luminance standard ITU-R BT.601 :
    ``Y = 0.299 R + 0.587 G + 0.114 B``.

    Parameters
    ----------
    image : npt.NDArray
        Tableau NumPy de shape ``(H, W)`` ou ``(H, W, 3)``.

    Returns
    -------
    npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``, même dtype que l'entrée.
    """
    if image.ndim == 2:
        return image
    return (
        0.299 * image[..., 0]
        + 0.587 * image[..., 1]
        + 0.114 * image[..., 2]
    ).astype(image.dtype)
