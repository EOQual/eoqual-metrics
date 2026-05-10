"""
Mesure de netteté Tenengrad basée sur les gradients de Sobel.

Référence
---------
Krotkov, E. *Focusing*. Int J Comput Vision 1, 223–237 (1987).
https://doi.org/10.1007/BF00127822
"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np
import numpy.typing as npt
from scipy import ndimage

try:
    from skimage.filters import threshold_otsu as _threshold_otsu
    _SKIMAGE = True
except ImportError:
    _SKIMAGE = False


def tenengrad(
    image: npt.NDArray,
    threshold: Optional[Union[float, str]] = "max_frac",
    thresh_fraction: float = 0.2,
    window: Optional[int] = None,
    normalize: bool = False,
    return_map: bool = False,
) -> Union[float, Tuple[float, npt.NDArray]]:
    """
    Calcule la mesure de netteté Tenengrad.

    L'algorithme applique des filtres de Sobel, calcule la carte d'énergie
    ``M = Gx² + Gy²``, applique un seuillage optionnel et retourne la somme
    (ou moyenne) des énergies au-dessus du seuil.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
    threshold : float or {'max_frac', 'mean', 'median', 'otsu'} or None, optional
        Méthode de seuillage appliquée à la carte d'énergie.

        * ``"max_frac"`` (défaut) : ``thresh_fraction * max(M)``
        * ``"mean"`` : ``thresh_fraction * mean(M)``
        * ``"median"`` : ``thresh_fraction * median(M)``
        * ``"otsu"`` : seuil d'Otsu (scikit-image requis)
        * ``None`` : aucun seuillage
        * ``float`` : seuil fixe
    thresh_fraction : float, optional
        Fraction utilisée avec ``"max_frac"``, ``"mean"``, ``"median"``.
        Par défaut ``0.2``.
    window : int or None, optional
        Taille du filtre uniforme de moyenne locale appliqué sur ``M``.
        Par défaut ``None`` (pas de filtrage).
    normalize : bool, optional
        Si ``True``, retourne la moyenne des valeurs seuillées (plutôt que
        leur somme). Par défaut ``False``.
    return_map : bool, optional
        Si ``True``, retourne aussi la carte d'énergie ``M``.
        Par défaut ``False``.

    Returns
    -------
    float
        Score Tenengrad. Valeur plus élevée = image plus nette.
    npt.NDArray, optional
        Carte d'énergie ``M`` de shape ``(H, W)``
        (seulement si ``return_map=True``).

    Raises
    ------
    ValueError
        Si ``threshold`` n'est pas reconnu.
    ImportError
        Si ``threshold='otsu'`` et scikit-image est absent.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(128, 128).astype(np.float32)
    >>> score = tenengrad(img)
    >>> score, M = tenengrad(img, return_map=True)
    """
    Gx = ndimage.sobel(image, axis=1)
    Gy = ndimage.sobel(image, axis=0)
    M: npt.NDArray = Gx ** 2 + Gy ** 2

    if window is not None and window > 1:
        M = ndimage.uniform_filter(M, size=window)

    if threshold is None:
        thresh_value: float = -np.inf
    elif isinstance(threshold, (int, float)):
        thresh_value = float(threshold)
    elif isinstance(threshold, str):
        s = threshold.lower()
        if s == "max_frac":
            thresh_value = thresh_fraction * float(np.max(M))
        elif s == "mean":
            thresh_value = thresh_fraction * float(np.mean(M))
        elif s == "median":
            thresh_value = thresh_fraction * float(np.median(M))
        elif s == "otsu":
            if not _SKIMAGE:
                raise ImportError("scikit-image est requis pour threshold='otsu'.")
            thresh_value = float(_threshold_otsu(M))
        else:
            raise ValueError(f"Méthode de seuillage inconnue : {threshold!r}")
    else:
        raise ValueError("'threshold' doit être un float, une str ou None.")

    mask = M > thresh_value
    score = float(np.mean(M[mask])) if normalize else float(np.sum(M[mask]))
    if not np.any(mask):
        score = 0.0

    return (score, M) if return_map else score
