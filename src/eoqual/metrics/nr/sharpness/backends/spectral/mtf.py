"""
Estimation de la MTF (Modulation Transfer Function) par profil de bord.

Cette implémentation simplifie le calcul de la MTF en extrayant un profil
d'intensité horizontal au centre d'un bloc, puis en mesurant la réponse
maximale de bord (dérivée maximale du profil). Elle sert d'indicateur rapide
de la résolution effective d'une image ou d'un bloc.

Note
----
Il s'agit d'une estimation grossière, non d'une MTF complète au sens
métrologique du terme. Pour une MTF rigoureuse, utiliser la méthode
slanted-edge (ISO 12233).
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from skimage.measure import profile_line


def mtf(
    image: npt.NDArray,
    block_size: int = 128,
) -> float:
    """
    Estime la netteté d'une image par la réponse maximale de bord (MTF simplifiée).

    L'image est découpée en blocs non-chevauchants de taille ``block_size``.
    Pour chaque bloc, un profil d'intensité horizontal central est extrait,
    sa dérivée absolue est calculée et le maximum est retenu. Le score final
    est la moyenne des maxima sur tous les blocs valides.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris de shape ``(H, W)``.
        Les valeurs sont converties en ``float32`` en interne.
    block_size : int, optional
        Taille des blocs carrés (en pixels). Par défaut ``128``.

    Returns
    -------
    float
        Score MTF moyen. Valeur plus élevée = transitions de bord plus nettes.
        Retourne ``0.0`` si aucun bloc valide n'est trouvé.

    Examples
    --------
    >>> import numpy as np
    >>> img = np.random.rand(256, 256).astype(np.float32)
    >>> score = mtf(img)
    """
    image = image.astype(np.float32)
    h, w = image.shape
    scores = []

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            bloc = image[i : i + block_size, j : j + block_size]
            if bloc.shape[0] < block_size or bloc.shape[1] < block_size:
                continue
            y = bloc.shape[0] // 2
            profile = profile_line(bloc, (y, 0), (y, bloc.shape[1] - 1))
            edge_response = np.abs(np.diff(profile))
            if edge_response.size > 0:
                scores.append(float(np.max(edge_response)))

    return float(np.mean(scores)) if scores else 0.0
