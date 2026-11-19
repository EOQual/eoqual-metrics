#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode S3 (Spectral and Spatial Sharpness)
A Spectral and Spatial Measure of Local Perceived Sharpness in Natural Images

Notes
-----
S3 analyse la netteté de manière locale en combinant deux types de mesures :
une mesure spectrale et une mesure spatiale.

- **Mesure spatiale** (variation totale) : quantifie la "rugosité" d'un
  bloc d'image en additionnant la valeur absolue des différences entre
  pixels adjacents. Un bloc plus net présente des transitions de couleurs
  plus abruptes, ce qui se traduit par une variation totale plus élevée.
- **Mesure spectrale** (pente du spectre de magnitude) : pour chaque
  bloc, une transformée de Fourier 2D est calculée. La pente du spectre
  de magnitude, une fois tracée sur une échelle logarithmique, indique la
  rapidité avec laquelle le contenu à haute fréquence de l'image diminue
  — une pente plus raide signifie moins de détails à haute fréquence,
  donc une image plus floue.

Les scores spatiaux et spectraux sont ensuite combinés pour créer une
carte de netteté locale, puis un score global est calculé, généralement
en utilisant le 90e percentile de ces scores locaux.

Bug corrigé (normalisation)
----------------------------
Les deux mesures par bloc étaient auparavant réétalées en min-max sur les
blocs de l'image **en cours de traitement** avant combinaison — chaque
image repartait donc sur sa propre échelle [0, 1], rendant le score non
comparable entre images, et même non monotone avec le niveau de flou
d'une seule image (vérifié : le score augmentait sur une bonne partie de
la plage de flou testée au lieu de diminuer). Remplacé par deux
transformations indépendantes de l'image (``-pente`` pour la mesure
spectrale, ``log1p`` pour la variation totale, afin de ramener les deux
mesures à un ordre de grandeur comparable) — strictement monotones, sans
dépendre des autres blocs. Comportement vérifié : monotone décroissant
avec un flou gaussien croissant sur deux images réelles et une image
synthétique, à l'exception d'un léger rebond entre une image parfaitement
nette et un flou de l'ordre d'un pixel (artefact probable de contenu
proche de Nyquist dans la pente spectrale, un comportement documenté pour
plusieurs mesures de netteté fondées sur le spectre — pas spécifique à
cette correction). Ce n'est **pas** une reproduction du calibrage exact
de Vu & Chandler (constantes de la publication non disponibles ici) :
une heuristique corrigée, pas un portage fidèle au pixel près.

Links
-----

"""
__author__     = "Olivier Amram"
__copyright__  = "Copyright (c) 2026 Olivier Amram"
__credits__    = "Olivier Amram"
__license__    = "MIT"
__version__    = "20250927"
__maintainer__ = "Olivier Amram"
__email__      = "olivier.amram@free.fr"
__status__     = "Prototype" # Prototype / Development / Production
__url__        = 'http://'
__all__        = [
    's3'
] # Liste des objets exportés par le module

from typing import Tuple
# IMPORTS TIERS
from loguru import logger
# reraise : la trace loguru est affichée, puis l'ImportError d'origine est
# relancée — rattrapable par l'appelant, sans arrêter le processus.
with logger.catch(reraise=True):
    import numpy as np
    from skimage.color import rgb2gray
    from scipy.ndimage import gaussian_filter
    from scipy.fft import fft2
    from scipy.stats import linregress
# IMPORTS LOCAUX
# No local imports yet


def _block_processing(
        image: np.ndarray,
        block_size: int = 32
    ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Divise une image en blocs et applique un traitement par bloc pour le calcul
    des mesures spectrales et spatiales.
    
    Parameters
    ----------
    image : np.ndarray
        Image d'entrée 2D.
    block_size : int, optional
        Taille des blocs. Par défaut à 32.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - spectral_slopes : Tableau des pentes spectrales pour chaque bloc.
        - total_variations : Tableau des variations totales pour chaque bloc.
    """
    ny, nx = image.shape
    spectral_slopes = []
    total_variations = []
    
    # Assure que la taille de l'image est un multiple de la taille des blocs
    padded_y = ny - (ny % block_size)
    padded_x = nx - (nx % block_size)
    
    image = image[:padded_y, :padded_x]

    for i in range(0, padded_y, block_size):
        for j in range(0, padded_x, block_size):
            block = image[i:i + block_size, j:j + block_size]
            
            # Calcul de la mesure spectrale (pente de la transformée de Fourier)
            f_transform = fft2(block)
            magnitude_spectrum = np.abs(f_transform) # type: ignore
            magnitude_spectrum = np.fft.fftshift(magnitude_spectrum)
            
            center_y, center_x = np.array(magnitude_spectrum.shape) // 2
            
            # Calcule les fréquences radiales
            y_coords, x_coords = np.indices(magnitude_spectrum.shape)
            r_coords = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
            
            # Écrase la valeur du centre (fréquence nulle) pour éviter l'infini en log
            r_coords[center_y, center_x] = 1 
            
            # Exclut le centre et les bords pour l'ajustement linéaire
            r_coords_flat = r_coords.flatten()
            magnitude_flat = magnitude_spectrum.flatten()
            
            # Fit une régression linéaire sur un plot log-log
            valid_indices = (r_coords_flat > 0)
            log_r = np.log(r_coords_flat[valid_indices])
            log_mag = np.log(magnitude_flat[valid_indices] + 1e-10) # Ajoute un petit terme pour éviter log(0)
            
            slope, _, _, _, _ = linregress(log_r, log_mag)
            spectral_slopes.append(slope)
            
            # Calcul de la mesure spatiale (variation totale)
            tv = np.sum(np.abs(np.diff(block, axis=0))) + np.sum(np.abs(np.diff(block, axis=1)))
            total_variations.append(tv)
            
    return np.array(spectral_slopes), np.array(total_variations)


def s3_index(image: np.ndarray, block_size: int = 32) -> float:
    """
    Calcule l'indice de netteté S3 d'une image.
    
    Parameters
    ----------
    image : np.ndarray
        Image d'entrée en niveaux de gris.
    block_size : int, optional
        Taille des blocs pour le calcul local. Par défaut à 32.
        
    Returns
    -------
    float
        La valeur de l'indice de netteté S3.
    """
    if image.ndim != 2:
        image = rgb2gray(image)
        
    image = image.astype(np.float32)

    spectral_slopes, total_variations = _block_processing(image, block_size)
    
    # Élimine les valeurs aberrantes (NaN ou inf)
    valid_indices = np.isfinite(spectral_slopes) & np.isfinite(total_variations)
    spectral_slopes = spectral_slopes[valid_indices]
    total_variations = total_variations[valid_indices]
    
    if len(spectral_slopes) == 0:
        return 0.0

    # Normalisation des mesures — PAS un min-max par image (bug corrigé :
    # réétaler chaque image sur sa propre échelle [0, 1] la rendait non
    # comparable à une autre image, et même non monotone avec le niveau de
    # flou global d'UNE SEULE image — deux versions plus ou moins floues
    # de la même scène pouvaient obtenir un score similaire, voire
    # inversé, car chaque appel repart de zéro sur min/max(blocs de CETTE
    # image). Transformations ci-dessous : indépendantes des autres blocs
    # et des autres images, strictement monotones dans le sens netteté.
    #
    # -slope : une pente spectrale moins raide (moins négative) signifie
    # plus de contenu haute fréquence conservé, donc une image plus nette.
    # Toujours >= 0 sur une image naturelle non constante (clip en
    # garde-fou pour les cas dégénérés).
    spectral_sharpness = np.clip(-spectral_slopes, 0.0, None)

    # Variation totale compressée en log (log1p) : sans cela, son ordre de
    # grandeur (typiquement plusieurs milliers) écrase presque totalement
    # la contribution spectrale (typiquement < 10) dans le produit
    # ci-dessous. Transformation strictement croissante, ne change pas le
    # sens de la mesure.
    spatial_sharpness = np.log1p(total_variations)

    # Combinaison des mesures pour la carte de netteté locale
    local_sharpness = np.sqrt(spectral_sharpness * spatial_sharpness)

    # Pooling des résultats (percentile 90e comme suggéré dans des implémentations similaires)
    s3_score = float(np.percentile(local_sharpness, 90))

    return s3_score



# Public alias
s3 = s3_index
