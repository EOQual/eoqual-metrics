#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase Coherence and Sharpness Metrics

G. Blanchet, L. Moisan, B. Rougé, "Measuring the global phase coherence of an image" (ICIP 2008).
G. Blanchet, L. Moisan, "An explicit Sharpness Index related to Global Phase Coherence" (ICASSP 2012).
A. Leclaire, L. Moisan, "Une Variante non Périodique du Sharpness Index" (hal-00837646 , version 1, 23-06-2013)
A. Leclaire, L. Moisan, "No-reference image quality assessment and blind deblurring with sharpness metrics exploiting
                         Fourier phase information", (Journal of Mathematical Imaging and Vision, 2015).

Links
-----
Based on the MATLAB code by Lionel Moisan
https://www.math.u-psud.fr/~moisan/soft/sharpness_index/
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
    'sharpness_index'
] # Liste des objets exportés par le module

# IMPORTS STANDARD
import sys
# IMPORTS TIERS
from loguru import logger
with logger.catch(onerror=lambda _: sys.exit(1)):
    import numpy as np
    from skimage.color import rgb2gray
    from scipy.ndimage import gaussian_filter
    # For new code, we recommend using the scipy.fft module, 
    # which provides more efficient FFT algorithms and a more flexible interface.”
    # https://numpy.org/doc/stable/reference/routines.fft.html
    from scipy.fft import fft2, ifft2
    from scipy.special import erfc
# IMPORTS LOCAUX
# No local imports yet

def _perdecomp(u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Décomposition périodique de l'image u en composantes p (périodique) et s (lisse).

    Notes
    -----
    La transformée de Fourier, telle qu'elle est utilisée pour les images, suppose que l'image est
    périodique et qu'elle se répète à l'infini dans toutes les directions. Cependant, les images réelles
    ne sont pas périodiques : la valeur des pixels au bord de l'image est généralement très différente
    de la valeur des pixels de l'autre côté. Sans correction, la transformée de Fourier interprète ce
    saut brusque comme une fréquence élevée. Cela crée des artefacts de bord qui faussent
    le calcul de la netteté en surestimant l'énergie du gradient. La fonction _perdecomp résout ce
    problème en séparant l'image en deux composantes : une partie périodique et une partie "lisse"
    qui contient le saut de bord. L'algorithme de netteté n'utilise que la partie périodique,
    garantissant que les calculs dans le domaine de Fourier sont précis.

    Parameters
    ----------
    u : np.ndarray
        Image d'entrée 2D.

    Returns
    -------
    p : np.ndarray
        Composante périodique de u.
    s : np.ndarray
        Composante lisse de u.
    """
    ny, nx = u.shape
    u = u.astype(np.float32)

    X = np.arange(nx)
    Y = np.arange(ny)
    v = np.zeros_like(u)

    # Calcul de v suivant la définition MATLAB
    v[0, :] = u[0, :] - u[-1, :]
    v[-1, :] = -v[0, :]
    v[:, 0] += u[:, 0] - u[:, -1]
    v[:, -1] -= u[:, 0] - u[:, -1]

    # Calcul des cosinus
    fx = np.cos(2 * np.pi * X / nx)
    fy = np.cos(2 * np.pi * Y / ny)

    fx = np.tile(fx, (ny, 1))
    fy = np.tile(fy[:, np.newaxis], (1, nx))

    fx[0, 0] = 0.0  # éviter la division par zéro

    # FFT2 de v
    v_hat = fft2(v)

    denom = 2 - fx - fy
    s = np.real(ifft2(v_hat * 0.5 / denom)) # type: ignore

    p = u - s

    return p, s


def _dequant(u: np.ndarray) -> np.ndarray:
    """
    Déquantification d'une image par translation (1/2, 1/2) dans le domaine de Fourier.

    Notes
    -----
    Les images numériques sont discrètes ; les pixels sont des points à des coordonnées entières.
    Cependant, l'algorithme original a été conçu pour des signaux continus. La déquantification est
    une opération qui "déplace" virtuellement le signal de l'image d'un demi-pas dans le domaine de
    Fourier. Cela corrige l'erreur de phase qui se produit lorsque l'on applique une transformée de
    Fourier à un signal discret au lieu d'un signal continu, ce qui est particulièrement important pour
    les calculs de gradients.

    Parameters
    ----------
    u : np.ndarray
        Image d'entrée 2D.

    Returns
    -------
    v : np.ndarray
        Image déquantifiée.
    """
    ny, nx = u.shape
    mx = nx // 2
    my = ny // 2

    # Construction des vecteurs Tx et Ty suivant la formule MATLAB
    # mod(mx:mx+nx-1, nx) en Python c'est (np.arange(mx, mx + nx) % nx)
    Tx = np.exp(-1j * np.pi / nx * ((np.arange(mx, mx + nx) % nx) - mx))
    Ty = np.exp(-1j * np.pi / ny * ((np.arange(my, my + ny) % ny) - my))

    # Produit extérieur Ty' * Tx (Ty transposé multiplié par Tx)
    T = np.outer(Ty, Tx)

    # FFT 2D, translation, puis IFFT 2D
    v = ifft2(fft2(u) * T)

    return np.real(v) # type: ignore


def _logerfc(x: np.ndarray) -> np.ndarray:
    """
    Calcul du logarithme de la fonction erreur complémentaire pour tout x.

    Pour x <= 20, calcule log(erfc(x)).
    Pour x > 20, utilise une approximation asymptotique pour éviter l'underflow.

    Parameters
    ----------
    x : np.ndarray
        Valeurs d'entrée (peuvent être scalaire, vecteur ou matrice).

    Returns
    -------
    y : np.ndarray
        log(erfc(x)) calculé pour chaque élément de x.
    """
    x = np.asarray(x)
    y = np.empty_like(x, dtype=np.float32)

    mask = x > 20
    # Cas x > 20 : approximation asymptotique
    if np.any(mask):
        X = x[mask]
        z = 1.0 / (X**2)
        s = np.ones_like(X)
        for k in range(8, 0, -1):
            s = 1 - (k - 0.5) * z * s
        y[mask] = -0.5 * np.log(np.pi) - X**2 + np.log(s / X)

    # Cas x <= 20 : calcul direct
    y[~mask] = np.log(erfc(x[~mask]))

    return y


# Fonction ω(t)
def _oomega(t):
    t = np.clip(t, -1, 1)
    return t * np.arcsin(t) + np.sqrt(1 - t**2) - 1


def sharpness_index(image: np.ndarray, pmode: int=3) -> float:
    """
    Permet de calculer l'indice de netteté (Sharpness Index) d'une image en niveaux de gris.
    Cet algorithme évalue la netteté d'une image en se basant sur la variation
    totale (Total Variation) et l'énergie du gradient. L'indice est plus faible
    pour les images floues et plus élevé pour les images nettes.

    Parameters
    ----------
    image : np.ndarray
        L'image d'entrée, représentée par un tableau NumPy 2D en niveaux de gris.
    pmode : int {0, 1, 2, 3}, optional
        Mode de prétraitement de l'image :
        0 : raw sharpness index of u
        1 : sharpness index of the periodic component of u 
        2 : sharpness index of the 1/2,1/2-translated of u 
        3 : sharpness index of the 1/2,1/2-translated of the periodic component of u
        Default mode (pmode = 3) should be used, unless you want to work on very
        specific images that are naturally periodic or not quantized (see paper)
        
    Returns
    -------
    float
        La valeur de l'indice de netteté.
    """
    if pmode == 1 or pmode == 3:
        image, _ = _perdecomp(image)
    if pmode == 2 or pmode == 3:
        image = _dequant(image)

    # Conversion en float32
    u = image.astype(np.float32)
    ny, nx = u.shape

    # Dérivées périodiques (différences circulaires)
    gx = u[:, np.r_[1:nx, 0]] - u
    gy = u[np.r_[1:ny, 0], :] - u
    fgx = fft2(gx)
    fgy = fft2(gy)

    # Variation totale (TV)
    tv = np.sum(np.abs(gx) + np.abs(gy))
    print(tv)

    # Calcul des produits de Fourier
    Gxx = np.real(ifft2(fgx * np.conj(fgx))) # type: ignore
    Gyy = np.real(ifft2(fgy * np.conj(fgy))) # type: ignore
    Gxy = np.real(ifft2(fgx * np.conj(fgy))) # type: ignore

    # Initialisation de la variance
    var = 0.0
    axx = Gxx[0, 0]
    ayy = Gyy[0, 0]
    if axx > 0:
        var += axx * np.sum(_oomega(Gxx / axx))
    if ayy > 0:
        var += ayy * np.sum(_oomega(Gyy / ayy))
    axy = np.sqrt(axx * ayy) if axx > 0 and ayy > 0 else 0
    if axy > 0:
        var += 2 * axy * np.sum(_oomega(Gxy / axy))

    var *= 2 / np.pi

    # Calcul de l'indice de netteté
    if var > 0:
        mu = (np.sqrt(axx) + np.sqrt(ayy)) * np.sqrt(2 * nx * ny / np.pi)
        t = (mu - tv) / np.sqrt(var)
        si = -_logerfc(t / np.sqrt(2)) / np.log(10) + np.log10(2)
        return si
    else:
        return 0.0

