"""
Backend commun pour les métriques perceptuelles calculées via ``piq``
(PyTorch Image Quality, Apache-2.0 — https://github.com/photosynthesis-team/piq).

Remplace, pour les métriques concernées, l'ancien backend ``IQA_pytorch``
(non maintenu depuis ~2021, cassé avec torch récent pour plusieurs
métriques — voir ``THIRD_PARTY_LICENSES.md`` et le commentaire dans
``config.py``). ``IQA_pytorch`` reste disponible en algo legacy là où
``piq`` n'a pas d'équivalent (``nlpd``, ``mad``) ou par compatibilité
ascendante.

Convention de tenseurs
-----------------------
``piq`` attend des tenseurs ``(N, C, H, W)`` dans ``[0, 1]``. Certaines
métriques (``DISTS``, ``LPIPS``, ``fsim`` avec ``chromatic=True``)
nécessitent un tenseur à 3 canaux même pour une image en niveaux de gris
(elles s'appuient sur un extracteur de features VGG16 pré-entraîné sur
ImageNet, donc RGB) — ce module réplique systématiquement le niveau de
gris sur 3 canaux pour éviter les cas particuliers par métrique.

Dépendance réseau
-----------------
``DISTS`` et ``LPIPS`` téléchargent les poids VGG16 pré-entraînés
(``vgg16-397923af.pth``, ~528 Mo) depuis ``download.pytorch.org`` au
premier appel, mis en cache ensuite dans
``~/.cache/torch/hub/checkpoints/``. Nécessite un accès réseau la
première fois ; sans connexion, l'appel échoue (``URLError``) — même
contrainte que l'ancien backend ``IQA_pytorch`` pour ces deux métriques
(déjà basées sur un backbone VGG pré-entraîné).
"""
from __future__ import annotations

from typing import Tuple

import numpy.typing as npt

try:
    import torch
    _TORCH = True
except ImportError:
    _TORCH = False

_device = None
_dists_model = None
_lpips_model = None


def piq_available() -> bool:
    """``True`` si ``torch`` et ``piq`` sont tous deux importables."""
    if not _TORCH:
        return False
    try:
        import piq  # noqa: F401
        return True
    except ImportError:
        return False


def _get_device() -> "torch.device":
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def to_rgb_tensors(GT: npt.NDArray, P: npt.NDArray) -> Tuple["torch.Tensor", "torch.Tensor"]:
    """
    Convertit une paire d'images en tenseurs ``(1, 3, H, W)`` dans
    ``[0, 1]``, sur le device disponible (GPU si présent).

    Les images en niveaux de gris sont répliquées sur 3 canaux (voir
    docstring du module) — nécessaire pour toutes les métriques ``piq``
    utilisées ici (``DISTS``/``LPIPS`` à cause de leur backbone VGG,
    ``fsim`` à cause de sa composante chromatique).

    Parameters
    ----------
    GT : npt.NDArray
        Image de référence (« target » dans le vocabulaire ``piq``).
    P : npt.NDArray
        Image traitée / dégradée (« predicted »).

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        ``(x, y)`` = (prédit, cible) — convention d'argument de ``piq``
        (``metric(x, y)``), à distinguer de la convention eoqual
        ``(GT, P)`` = (référence, traité).

    Raises
    ------
    ImportError
        Si ``torch`` n'est pas installé.
    """
    if not _TORCH:
        raise ImportError("PyTorch est requis : pip install \"eoqual-metrics[deep]\".")
    from ...core.image_utils import prepare_image

    device = _get_device()
    ref = prepare_image(GT).to(device)
    dist = prepare_image(P).to(device)
    if ref.shape[1] == 1:
        ref = ref.repeat(1, 3, 1, 1)
        dist = dist.repeat(1, 3, 1, 1)
    return dist, ref  # (x=predicted, y=target)


def get_dists_model():
    """Instance mise en cache de ``piq.DISTS`` (évite de recharger VGG16 à chaque appel)."""
    global _dists_model
    if _dists_model is None:
        import piq
        _dists_model = piq.DISTS().to(_get_device())
    return _dists_model


def get_lpips_model():
    """Instance mise en cache de ``piq.LPIPS`` (backbone VGG16, voir docstring du module)."""
    global _lpips_model
    if _lpips_model is None:
        import piq
        _lpips_model = piq.LPIPS().to(_get_device())
    return _lpips_model
