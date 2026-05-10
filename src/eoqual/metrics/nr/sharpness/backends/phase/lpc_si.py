#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Méthode LPC-SI (Local Phase Coherence - Sharpness Index).
Image Sharpness Assessment Based on Local Phase Coherence

Notes
-----
La netteté perçue par l'œil humain est étroitement liée à la phase des fréquences spatiales de l'image.
Les images nettes ont une forte cohérence de phase le long des contours, tandis que le flou la réduit.
L'algorithme décompose l'image en différentes échelles et orientations à l'aide de filtres de Log-Gabor
pour évaluer cette cohérence de phase.

Principe de la méthode :
1. Décomposition multi-échelle
    - L’image est transformée (typiquement par complex wavelet transform).
    - Chaque pixel est représenté par amplitude + phase à plusieurs échelles.
2. Analyse de la cohérence de phase
    - En bord net → les composantes d’échelles différentes s’alignent en phase.
    - En bord flou → la cohérence de phase se dégrade.
3. Mesure de la cohérence locale
    - Pour chaque position, on calcule un indice de phase congruency ou phase coherence.
    - C’est essentiellement une mesure de stabilité des phases entre échelles.
4. Agrégation globale
    - On combine ces indices locaux en un score de netteté global.
    - Un score élevé = image jugée nette.

Links
-----
Porté depuis le code MATLAB original de Peter Kovesi :
https://ece.uwaterloo.ca/~z70wang/research/lpcsi/
"""
__author__     = "Olivier Amram"
__copyright__  = ""
__credits__    = "Olivier Amram"
__license__    = ""
__version__    = "20251018"
__maintainer__ = "Olivier Amram"
__email__      = "olivier.amram@free.fr"
__status__     = "Prototype" # Prototype / Development / Production
__url__        = 'http://'
__all__        = [
    'lpc_si'
] # Liste des objets exportés par le module

# IMPORTS STANDARD
import sys
from typing import List, Tuple, Union
# IMPORTS TIERS
from loguru import logger
with logger.catch(onerror=lambda _: sys.exit(1)):
    import numpy as np
    from skimage.color import rgb2gray
    from scipy.ndimage import gaussian_filter
    from scipy.fft import fft2, ifft2, fftshift, ifftshift
    from scipy.stats import gmean
# No local imports


def _lowpassfilter(
        sze: Tuple[int, int],
        cutoff: float,
        n: int
    ) -> np.ndarray:
    """
    Construit un filtre passe-bas de Butterworth.

    Parameters
    ----------
    sze : Tuple[int, int]
        Un tuple de deux éléments spécifiant la taille du filtre à construire [lignes, colonnes].
    cutoff : float
        La fréquence de coupure du filtre (0 - 0.5).
    n : int
        L'ordre du filtre, plus n est grand, plus la transition est nette. n doit être un entier >= 1.

    Returns
    -------
    np.ndarray
        Le filtre passe-bas 2D.
    
    Notes
    -----
    Le filtre de Butterworth est défini par la formule :
        H(u,v) = 1 / (1 + (w / cutoff)^(2n))

    Raises
    ------
    ValueError
        Si la fréquence de coupure n'est pas dans l'intervalle [0, 0.5] ou si n n'est pas un entier >= 1.
    """
    if cutoff < 0 or cutoff > 0.5:
        logger.error("La fréquence de coupure doit être entre 0 et 0.5")
        raise ValueError("La fréquence de coupure doit être entre 0 et 0.5")
    if n % 1 != 0 or n < 1:
        logger.error("n doit être un entier >= 1")
        raise ValueError("n doit être un entier >= 1")

    # Dimensions du filtre
    rows, cols = sze
    
    # Créer les plages xrane et yrange normalisées à +/- 0.5
    # Ajustement pour les tailles paires et impaires
    if cols % 2 == 1:
        xrange = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols) / (cols - 1)
    else:
        xrange = np.linspace(-cols / 2, cols / 2 - 1, cols) / cols
        
    if rows % 2 == 1:
        yrange = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows) / (rows - 1)
    else:
        yrange = np.linspace(-rows / 2, rows / 2 - 1, rows) / rows

    # Créer les grilles x et y
    x, y = np.meshgrid(xrange, yrange)
    
    # Calculer le rayon relatif au centre
    radius = np.sqrt(x**2 + y**2)
    radius[0, 0] = 1 # Évite la division par zéro

    # Calculer le filtre passe-bas
    f = 1.0 / (1.0 + (radius / cutoff)**(2 * n))
    return fftshift(f)


def _logGabor_2D(
        im_shape: Tuple[int, int],
        norient: int,
        nscale: int,
        scalefac: Union[float, List[float], np.ndarray],
        sigma: float
    ) -> List[List[np.ndarray]]:
    """
    Construit un banc de filtres de Log-Gabor en 2D.

    Parameters
    ----------
    im_shape : Tuple[int, int]
        La taille de l'image pour laquelle les filtres sont construits [lignes, colonnes].
    norient : int
        Le nombre d'orientations des filtres.
    nscale : int
        Le nombre d'échelles des filtres.
    scalefac : float or List[float] or np.ndarray
        Le facteur d'échelle entre les filtres successifs ou une liste de facteurs d'échelle.
    sigma : float
        Le rapport de l'écart-type de la gaussienne décrivant la fonction de transfert du filtre
        de Log-Gabor dans le domaine fréquentiel à la fréquence centrale du filtre.

    Returns
    -------
    List[List[np.ndarray]]
        Une liste de listes contenant les filtres de Log-Gaborpour chaque échelle et orientation.
    """
    # Paramètres du filtre
    dThetaOnSigma = 1.5     # Rapport de l'intervalle angulaire à l'écart-type angulaire
    minWaveLength = 4       # Longueur d'onde du filtre à la plus petite échelle

    # Taille de l'image
    rows, cols = im_shape
    
    # Calcul de l'écart-type de la fonction Gaussienne angulaire
    thetaSigma = np.pi / norient / dThetaOnSigma
    
    # Créer les plages xrane et yrange normalisées à +/- 0.5
    # Ajustement pour les tailles paires et impaires
    if cols % 2 == 1:
        xrange = np.linspace(-(cols - 1) / 2, (cols - 1) / 2, cols) / (cols - 1)
    else:
        xrange = np.linspace(-cols / 2, cols / 2 - 1, cols) / cols
        
    if rows % 2 == 1:
        yrange = np.linspace(-(rows - 1) / 2, (rows - 1) / 2, rows) / (rows - 1)
    else:
        yrange = np.linspace(-rows / 2, rows / 2 - 1, rows) / rows
    
    # Créer les grilles x et y
    x, y = np.meshgrid(xrange, yrange)
    del xrange, yrange # Libère la mémoire

    # Calculer le rayon et l'angle polaire
    radius = np.sqrt(x**2 + y**2)   # Rayon normalisé par rapport au centre
    theta = np.arctan2(-y, x)       # Angle polaire (sens anti-horaire positif)

    # Décalage des quadrants pour placer la fréquence 0 aux coins
    radius = ifftshift(radius)
    theta  = ifftshift(theta)
    radius[0, 0] = 1  # Éviter la division par zéro pour le log    

    # Calculer sinus et cosinus pour les différences angulaires
    sintheta = np.sin(theta)
    costheta = np.cos(theta)
    del x, y, theta # Libère la mémoire

    # Créer le filtre passe-bas pour limiter les fréquences aux coins
    lp = _lowpassfilter(im_shape, 0.45, 15) # Radius .45, 'sharpness' 15

    # Initialiser les composants radiaux (log-Gabor)
    logGabor_filt = [None] * nscale
    
    for s in range(nscale):
        # Calculer la longueur d'onde pour l'échelle courante
        if isinstance(scalefac, (list, np.ndarray)):
            wavelength = minWaveLength * scalefac[s]
        else:
            wavelength = minWaveLength * (scalefac ** s)
        
        # Fréquence centrale du filtre       
        fo = 1.0 / wavelength

        # Calculer le composant radial Log-Gabor
        logGabor_filt[s]        = np.exp((-(np.log(radius/fo)**2)) / (2 * np.log(sigma)**2))
        logGabor_filt[s]       *= lp    # Appliquer le filtre passe-bas  # type: ignore
        logGabor_filt[s][0, 0]  = 0     # Remettre la fréquence 0 à zéro # type: ignore
    
    # Initialiser les composants angulaires
    spread = [None] * norient

    for o in range(norient):
        # Angle du filtre
        angl = (o) * np.pi / norient

        # Calculer la distance angulaire
        ds = sintheta * np.cos(angl) - costheta * np.sin(angl)  # Différence en sinus
        dc = costheta * np.cos(angl) + sintheta * np.sin(angl)  # Différence en cosinus
        dtheta = np.abs(np.arctan2(ds, dc))                     # Distance angulaire absolue

        # Calculer le composant angulaire
        spread[o] = np.exp(-(dtheta**2) / (2 * thetaSigma**2))

    # Boucle principale pour construire les filtres
    filter = [[None] * norient for _ in range(nscale)]
    for o in range(norient):
        for s in range(nscale):
            # Combiner les composants radial et angulaire
            filter[s][o] = logGabor_filt[s] * spread[o] # type: ignore

    return filter # type: ignore


def lpc_si(
        im: np.ndarray,
        C: int=2,
        Beta_k: float=1e-4, 
        scales: list=[1, 3/2, 2],
        norient: int=8,
        B: int=None, # type: ignore
        map: bool=False
    ) -> Tuple[float, np.ndarray]:
    """
    Calcule l'indice de netteté basé sur la cohérence de phase locale.

    Parameters
    ----------
    im : np.ndarray
        L'image d'entrée, représentée par un tableau NumPy 2D en niveaux de gris.
    C : int, optional
        Un paramètre de la formule. La valeur par défaut est 2.
    Beta_k : float, optional
        Un petit terme pour éviter la division par zéro. La valeur par défaut est 1e-4.
    scales : List[float], optional
        Une liste des échelles des filtres de Log-Gabor. La valeur par défaut est [1, 3/2, 2].
    norient : int, optional
        Le nombre d'orientations des filtres. La valeur par défaut est 8.
    B : int, optional
        La taille du bloc pour le calcul local. Si None, elle est calculée par défaut.
    map : bool, optional
        Si True, retourne également la carte de cohérence de phase locale. La valeur par défaut est False.

    Returns
    -------
    Tuple[float, np.ndarray]
        - si : La valeur de l'indice de netteté.
        - lpc_map : La carte de cohérence de phase locale.

    """
    rows, cols = im.shape
    nscale     = len(scales)
    w          = np.array([1, -3, 2])   # Poids associés aux échelles
                                        # Correspond à la configuration scales = [1, 3/2, 2]
    if B is None:
        B = int(np.round(min(rows, cols) / 16))

    # Générer les filtres Log-Gabor
    filters = _logGabor_2D(im.shape, norient, nscale, scales, 0.33)

    # Calculer la transformée de Fourier de l'image
    im_fft = fft2(im)

    # Initiliaser les tableaux
    s_lpc = np.ones((rows, cols, norient))
    energy = np.zeros((rows, cols, norient))
    M = np.zeros((rows, cols, nscale), dtype=complex)

    # Boucle sur les orientations et échelles
    for o in range(norient):
        for s in range(nscale):
            # Appliquer le filtre dans le domaine fréquentiel et revenir dans le domaine spatial
            M[:, :, s] = ifft2(im_fft * filters[s][o])
            # np.abs : Magnitude pour éviter les problèmes de type complexe
            s_lpc[:, :, o] *= np.abs(M[:, :, s] ** w[s])
        
        # Calculer l'énergie pour la première échelle
        e = np.abs(M[:, :, 0])
        e_center = e[B:-B, B:-B]
        e_mean = np.mean(e_center)
        e_std = np.std(e_center)
        T = e_mean + 2 * e_std
        e = np.maximum(0, e - T)
        energy[:, :, o] = e

    # Calculer la carte LPC
    s_lpc_map = np.cos(np.angle(s_lpc))
    #print(">", np.mean(s_lpc_map[:,:,0]))
    s_lpc_map[s_lpc_map < 0] = 0
    lpc_map = np.sum(s_lpc_map * energy, axis=2) / (np.sum(energy, axis=2) + C)

    # Extraire la région centrale
    lpc_map_center = lpc_map[B:-B, B:-B]

    # Calculer l'indice de saillance (si)
    sorted_si = np.sort(lpc_map_center.ravel())[::-1]  # Tri décroissant
    N = len(sorted_si)
    u = np.exp(-np.arange(N) / (N - 1) / Beta_k)
    si = np.sum(sorted_si * u) / np.sum(u)

    # Retourner en fonction du paramètre map
    if map:
        return si, lpc_map
    return si
