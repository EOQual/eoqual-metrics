#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode PSI (Perceptual Sharpness Index).

Étapes principales de l'algorithme :
1. Détection des contours (Sobel).
    - Les zones de contour sont les candidates pour évaluer la netteté.
2. Analyse des gradients autour des contours.
    - On calcule la direction (angle) du gradient en chaque point.
    - On cherche spécifiquement les bords horizontaux/verticaux dominants (les plus perceptuellement importants).
3. Mesure des largeurs de transitions.
    - Pour un bord, on regarde combien de pixels il faut pour passer d’un minimum local à un maximum local
    (transition sombre → clair ou l’inverse).
    - Plus cette transition est courte → plus le bord est net.
    - On applique une correction (JNB, Just Noticeable Blur) pour enlever les transitions qui sont trop fines pour être perceptibles.
4. Agrégation en blocs.
    - L’image est découpée en blocs (par défaut 32×32).
    - Pour chaque bloc, on calcule la largeur moyenne des transitions.
    - Seuls les blocs les plus nets (par ex. les 22% supérieurs) sont pris en compte, 
    car la netteté perceptuelle dépend surtout des zones les plus contrastées.
5.Score final.
    - Défini comme le nombre de bords nets retenus divisé par la somme de leurs largeurs.
    - Plus le score est grand → plus l’image est perçue comme nette.

Notes
-----
- L’image n’est pas analysée globalement, mais localement (par blocs).
- Le percentile (par défaut 22%) permet de se concentrer uniquement sur les zones nettes.
- Le Just Noticeable Blur (JNB) est une correction perceptuelle importante pour éviter les bordures trop épaisses.
- Contrairement à d'autres méthodes, elle ne dépend pas de la fréquence mais de la structure des contours.
- Il est un peu plus lent à cause du traitement par bloc + boucles.
- Il ne considère que les contours verticaux (±90°), donc moins sensible aux contours diagonaux ou horizontaux.
- Pas robuste au bruit si mal pré-traité (ex. si l’image est très bruitée, beaucoup de contours erronés sont pris en compte).

Links
-----
Based on the MATLAB code by Feichtenhofer et al. (2013)
https://feichtenhofer.github.io/
https://github.com/feichtenhofer/PSI
"""
__author__     = "Olivier Amram"
__copyright__  = ""
__credits__    = "Olivier Amram"
__license__    = ""
__version__    = "20250927"
__maintainer__ = "Olivier Amram"
__email__      = "olivier.amram@free.fr"
__status__     = "Prototype" # Prototype / Development / Production
__url__        = 'http://'
__all__        = [
    'psi'
] # Liste des objets exportés par le module

# IMPORTS STANDARD
import sys
# IMPORTS TIERS
from loguru import logger
with logger.catch(onerror=lambda _: sys.exit(1)):
    import numpy as np
    from skimage.color import rgb2gray
    from scipy.ndimage import gaussian_filter
    from skimage import color, feature    
# IMPORTS LOCAUX
# No local imports yet


def psi(I: np.ndarray, percentile: int = 22) -> float:
    """
    Compute the perceptual sharpness (PSI) of an image.

    This function implements the perceptual image sharpness metric based on
    local edge gradient analysis as described in:

        C. Feichtenhofer, H. Fassold, P. Schallauer,
        "A perceptual image sharpness metric based on local edge gradient
        analysis", IEEE Signal Processing Letters, 20 (4), 379-382, 2013.

    Parameters
    ----------
    I : np.ndarray
        Input image, either grayscale or RGB. If RGB, it is converted to grayscale.
    percentile : int, optional
        Percentage of blocks to use for the sharpness metric. Default is 22.

    Returns
    -------
    float
        The perceptual sharpness score of the image.
    """
    # Parameters
    BLOCKSIZE = 32        # Size for averaging of width measurements
    THRESHOLD_W = 2       # Minimum sum of widths in block
    QUOTA_W = percentile / 100.0  # Fraction of blocks used

    # Sobel edge detection
    edges = feature.canny(I, sigma=1)

    m, n = I.shape
    edge_widths = np.zeros((m, n))
    widths_count = 0

    # Compute image gradients (approximation of MATLAB gradient)
    Ix = np.zeros_like(I)
    Iy = np.zeros_like(I)

    Ix[:, 0] = I[:, 1] - I[:, 0]
    Ix[:, -1] = I[:, -1] - I[:, -2]
    Ix[:, 1:-1] = 0.5 * (I[:, 2:] - I[:, :-2])

    Iy[0, :] = I[1, :] - I[0, :]
    Iy[-1, :] = I[-1, :] - I[-2, :]
    Iy[1:-1, :] = 0.5 * (I[2:, :] - I[:-2, :])

    # Gradient angle in degrees
    phi = np.degrees(np.arctan2(Iy, Ix))

    # Parameters for edge width calculation
    t = 8       # Angle tolerance
    w_JNB = 3   # Just noticeable blur threshold

    row_idx, col_idx = np.where(edges)

    # Iterate over edge pixels
    for i, j in zip(row_idx, col_idx):
        if Ix[i, j] == 0 and Iy[i, j] == 0:
            continue

        # Horizontal edge (gradient upwards)
        if abs(phi[i, j] + 90) < t:
            width_up, width_down = 0, 0
            min_val, max_val = 0, 0

            # Search upwards
            for d in range(1, m):
                up = i - d
                if up < 0:
                    width_up = -1
                    break
                if I[up, j] <= I[up + 1, j]:
                    width_up = d - 1
                    max_val = I[up + 1, j]
                    break

            # Search downwards
            for d in range(1, m):
                down = i + d
                if down >= m:
                    width_down = -1
                    break
                if I[down, j] >= I[down - 1, j]:
                    width_down = d - 1
                    min_val = I[down - 1, j]
                    break

            if width_up != -1 and width_down != -1:
                widths_count += 1
                phi2 = np.radians(phi[i, j] + 90)
                edge_widths[i, j] = (width_up + width_down) / np.cos(phi2)
                slope = (max_val - min_val) / edge_widths[i, j]
                if edge_widths[i, j] >= w_JNB:
                    edge_widths[i, j] -= slope

        # Horizontal edge (gradient downwards)
        if abs(phi[i, j] - 90) < t:
            width_up, width_down = 0, 0
            min_val, max_val = 0, 0

            # Search upwards
            for d in range(1, m):
                up = i - d
                if up < 0:
                    width_up = -1
                    break
                if I[up, j] >= I[up + 1, j]:
                    width_up = d - 1
                    min_val = I[up + 1, j]
                    break

            # Search downwards
            for d in range(1, m):
                down = i + d
                if down >= m:
                    width_down = -1
                    break
                if I[down, j] <= I[down - 1, j]:
                    width_down = d - 1
                    max_val = I[down - 1, j]
                    break

            if width_up != -1 and width_down != -1:
                widths_count += 1
                phi2 = np.radians(phi[i, j] - 90)
                edge_widths[i, j] = (width_up + width_down) / np.cos(phi2)
                slope = (max_val - min_val) / (edge_widths[i, j] + 1e-12)
                if edge_widths[i, j] >= w_JNB:
                    edge_widths[i, j] -= slope

    # Average edge widths per block
    row_blocks = m // BLOCKSIZE
    col_blocks = n // BLOCKSIZE
    avg_w = []

    for i in range(1, row_blocks - 1):
        for j in range(1, col_blocks - 1):
            block_row = i * BLOCKSIZE
            block_col = j * BLOCKSIZE
            block_widths = edge_widths[block_row:block_row + BLOCKSIZE,
                                       block_col:block_col + BLOCKSIZE]
            w_sum = np.sum(block_widths)
            if w_sum >= THRESHOLD_W:
                avg_w.append(w_sum / np.count_nonzero(block_widths))

    avg_w = np.array(avg_w)
    nr_of_used_blocks = int(np.ceil(len(avg_w) * QUOTA_W))

    if nr_of_used_blocks == 0 or widths_count == 0:
        return 0.0

    avg_sorted = np.sort(avg_w)
    sharpest_edges = avg_sorted[:nr_of_used_blocks]

    return len(sharpest_edges) / np.sum(sharpest_edges)

