from typing import List, Union

import numpy as np
import numpy.typing as npt
from scipy.ndimage import convolve
from scipy.fftpack import dctn
from scipy.ndimage import generic_filter
#from skimage.measure import shannon_entropy
import pywt


__all__ = [
    "MEASURES_FOCUS",
    "fmeasure"
]

MEASURES_FOCUS = [
    "ACMO", "BREN", "CONT", "CURV", "DCTE",
    "DCTR", "GDER", "GLLV", "GLVA", "GLVN",
    "GRAE", "GRAS", "GRAT", "HELM", 
    "HISR", "LAPD", "LAPE", "LAPM", "LAPV",
    "SFIL", "SFRQ", "TENG", "TENV", "VOLA",
    "WAVR", "WAVS", "WAVV"
]

_BAD = [
    "HISE",
]

def fmeasure(Image: npt.NDArray, algo: str, ROI: Union[List[int], None] = None) -> float:
    """
    fmeasure: This function measures the relative degree of focus of an image.
    
    Inspired by the MATLAB implementation of the same name available at:
    [https://fr.mathworks.com/matlabcentral/fileexchange/27314-focus-measure]

    Parameters
    ----------
    Image : npt.NDArray
        Image is a grayscale image and FM is the computed
    Measure : str
        Measure is the focus measure algorithm as a string.
        see 'operators.txt' for a list of focus measure methods.
    ROI : List(int)
        Image ROI as a rectangle [xo yo width heigth].
        if an empty argument is passed, the whole image is processed.

    Returns
    -------
    float
        Relative degree of focus measurement

    Raises
    ------
    ValueError
        _description_

    [https://sites.google.com/view/cvia/focus-measure]

    Implemented focus measure operators:
    -----------------------------------
    ACMO: Absolute central moment (Shirvaikar2004)
    BREN: Brenner's focus measure (Santos97)
    CONT: Image contrast (Nanda2001)
    CURV: Image curvature (Helmli2001)
    DCTE: DCT Energy measure (Shen2006)
    DCTR: DCT Energy ratio (Lee2009)
    GDER: Gaussian derivative (Geusebroek2000)
    GLLV: Gray-level local variance (Pech2000)
    GLVA: Gray-level variance (Krotkov86)
    GLVN: Gray-level variance normalized (Santos97)
    GRAE: Energy of gradient (Subbarao92)
    GRAS: Squared gradient (Eskicioglu95)
    GRAT: Thresholded gradient (Santos97)
    HELM: Helmli's measure (Helmli2001)
#    HISE: Histogram entropy (Krotkov86)
    HISR: Histogram range (Firestone91)
    LAPD: Diagonal Laplacian (Thelen2009)
    LAPE: Energy of Laplacian (Subbarao92)
    LAPM: Modified laplacian (Nayar89)
    LAPV: Variance of laplacian (Pech2000)
    SFIL: Steerable filters-based (Minhas2009)
    SFRQ: Spatial frequency (Eskicioglu95)
    TENG: Tenegrad (Krotkov86)
    TENV: Tenengrad variance (Pech2000)
    VOLA: Vollat's correlation-based (Santos97)
    WAVR: Wavelet ratio (Xie2006)
    WAVS: Wavelet sum (Yang2003)
    WAVV: Wavelet variance (Yang2003)

    Note: The implementations provided in this package are those of 
    the author based on his interpretation of the original papers.
    For futher details, refer to:
    S. Pertuz et al., Analysis of focus measure operators for Shape-
    from-focus. Pattern Recognition, 46(5):1415-1432, 2013.
    DOI:10.1016/j.patcog.2012.11.011

    References:
    [Eskicioglu95] 	Eskicioglu, A. M & Fisher, P. S.
                    Image quality measures and their performance. 1995. 
    [Firestone91] 	Firestone, L.; Cook, K.; Culp, K.; Talsania, N. & Jr., K. P.
                    Comparison of autofocus methods for automated microscopy. 1991. 
    [Geusebroek2000]Geusebroek, J.; Cornelissen, F.; Smeilders, A. & Geerts, H
                    Robust autofocusing in microscopy. 2000.  
    [Helmli2001] 	Helmli, F. & Scherer, S.
                    Adaptive shape from focus with an error estimation in light microscopy. 2001.
    [Krotkov86] 	Krotkov, E.
                    Range from focus. 1986.
    [Lee2009] 		Sang-Yong Lee, Jae-Tack Yoo, K. Y. S. K
                    Reduced Energy-Ratio Measure for Robust Autofocusing in Digital Camera. 2009.
    [Minhas2009] 	Minhas, R.; Mohammed, A. A.; Wu, Q. M. & Sid-Ahmed, M. A.
                    3D Shape from Focus and Depth Map Computation Using Steerable Filter. 2009.
    [Nanda2001] 	Nanda, H. & Cutler, R.
                    Practical calibrations for a real-time digital omnidirectional camera. 2001. 
    [Nayar89] 		Nayar, S. K.
                    Tech. report: Shape from focus. 1989. 
    [Pech2000] 		Pech, J.; Cristobal, G.; Chamorro, J. & Fernandez, J.
                    Diatom autofocusing in brightfield microscopy: a comparative study. 2000. 
    [Santos97] 		Santos, A.; de Solorzano, C. O.; Vaquero, J. J.; Peña, J. M.; Mapica, N. & Pozo, F. D.
                    Evaluation of autofocus functions in moleclar cytogenetic analysis. 1997.
    [Shen2006] 		Shen, C. & Chen, H.
                    Robust focus measure for low-contrast images. 2006.
    [Shirvaikar2004]Shirvaikar, M.V.
                    An optimal measure for camera focus and exposure. 2004. 
    [Subbarao92] 	Subbarao, M.; Choi, T. & Nizkad, A.
                    Tech. report: Focusing techniques. 1992. 
    [Thelen2009] 	Thelen, A.; Frey, S.; Hirsch, S. & Hering, P. Interpolation
                    Improvements in Shape-From-Focus for Holographic Reconstructions With Regard to Focus Operators, Neighborhood-Size, and Height Value. 2009
    [Xie2006] 		Xie, H.; Rong, W. & Sun, L.
                    Wavelet-Based Focus Measure and 3-D Surface Reconstruction Method for Microscopy Images. 2006. 
    [Yang2003] 		Yang, G. & Nelson, B.
                    Wavelet-based autofocusing and unsupervised segmentation of microscopic images. 2003. 
    """
    if len(Image.shape) > 2:
        raise ValueError("L'image doit être en niveaux de gris !")

    if ROI is not None and len(ROI) > 0:
        Image = Image[ROI[1]:ROI[1]+ROI[3], ROI[0]:ROI[0]+ROI[2]]

    Image = Image.astype(np.float64)

    WSize = 15  # Size of local window (only some operators)

    algo = algo.upper()

    if algo == 'ACMO':  # Absolute Central Moment (Shirvaikar2004)
        #TODO: Pourquoi on force uint8 ?
        #if not np.issubdtype(Image.dtype, np.integer):
        #    Image = Image.astype(np.uint8)

        M, N = Image.shape
        Hist = np.histogram(Image.flatten(), bins=256)[0] / (M * N)
        Hist = np.abs(np.arange(256) - np.mean(Image)) * Hist
        FM = np.sum(Hist)

    elif algo == 'BREN':  # Brenner's (Santos97)
        DH = np.zeros_like(Image)
        DV = np.zeros_like(Image)
        DV[:-2, :] = Image[2:, :] - Image[:-2, :]
        DH[:, :-2] = Image[:, 2:] - Image[:, :-2]
        FM = np.maximum(DH, DV)
        FM = FM ** 2
        FM = np.mean(FM)

    elif algo == 'CONT':  # Image contrast (Nanda2001)
        def _ImContrast(x):
            return np.sum(np.abs(x - x[4]))

        # Appliquer la fonction ImContrast à chaque pixel avec une fenêtre 3x3
        FM = generic_filter(Image, _ImContrast, size=3)
        # Calculer la moyenne des valeurs obtenues pour toute l'image
        FM = np.mean(FM) # type: ignore

    elif algo == 'CURV':  # Image Curvature (Helmli2001)
        #TODO: Pourquoi on force uint8 ?
        #if not np.issubdtype(Image.dtype, np.integer):
        #    Image = Image.astype(np.uint8)

        # Définition des filtres M1 et M2
        M1 = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        M2 = np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]])

        # Calcul des différentes réponses de filtres
        mode = 'nearest' # 'constant'
        P0 = convolve(Image, M1, mode=mode) / 6 # type: ignore
        P1 = convolve(Image, M1.T, mode=mode) / 6 # type: ignore
        P2 = 3 * convolve(Image, M2, mode=mode) / 10 - convolve(Image, M2.T, mode='nearest') / 5 # type: ignore
        P3 = -convolve(Image, M2, mode=mode) / 5 + 3 * convolve(Image, M2, mode='nearest') / 10 # type: ignore

        # Calcul de la mesure caractéristique
        FM = np.abs(P0) + np.abs(P1) + np.abs(P2) + np.abs(P3)
        FM = np.mean(FM)

    elif algo == 'DCTE':  # DCT energy ratio (Shen2006)
        def _DctRatio(M):
            MT = dctn(M, type=2, norm='ortho')**2
            FM = (np.sum(MT) - MT[0, 0]) / MT[0, 0]
            return FM

        FM = [_DctRatio(Image[i:i+8, j:j+8]) for i in range(0, Image.shape[0], 8) for j in range(0, Image.shape[1], 8)]
        FM = np.mean(FM) # type: ignore

    elif algo == 'DCTR':  # DCT reduced energy ratio (Lee2009)
        def _ReRatio(M):
            M = dctn(M, type=2, norm='ortho')
            FM = (M[0, 1]**2 + M[0, 2]**2 + M[1, 0]**2 + M[1, 1]**2 + M[2, 0]**2) / (M[0, 0]**2)
            return FM

        FM = [_ReRatio(Image[i:i+8, j:j+8]) for i in range(0, Image.shape[0], 8) for j in range(0, Image.shape[1], 8)]
        #FM = generic_filter(Image, _ReRatio, size=8, mode='nearest')
        FM = np.mean(FM) # type: ignore

    elif algo == 'GDER':  # Gaussian derivative (Geusebroek2000)
        N = WSize // 2
        sig = N / 2.5
        x, y = np.meshgrid(np.arange(-N, N+1), np.arange(-N, N+1))
        G = np.exp(-(x**2 + y**2) / (2 * sig**2)) / (2 * np.pi * sig)
        Gx = -x * G / (sig**2)
        Gx /= np.sum(np.abs(Gx))
        Gy = -y * G / (sig**2)
        Gy /= np.sum(np.abs(Gy))
        Rx = convolve(Image, Gx, mode='nearest')
        Ry = convolve(Image, Gy, mode='nearest')
        FM = Rx**2 + Ry**2 # type: ignore
        FM = np.mean(FM)

    elif algo == 'GLLV':  # Graylevel local variance (Pech2000)
        # Calcul de la variance locale
        LVar = generic_filter(Image, np.var, size=(WSize, WSize))
        # Calcul de la mesure de caractéristique
        FM = np.var(LVar) # type: ignore

    elif algo == 'GLVA':  # Graylevel variance (Krotkov86)
        FM = np.std(Image)

    elif algo == 'GLVN':  # Normalized GLV (Santos97)
        FM = np.std(Image) ** 2 / np.mean(Image) # type: ignore

    elif algo == 'GRAE2':  # Energy of gradient (Subbarao92a)
        Ix = np.diff(Image, axis=1)[:-1, :] # Ajustement de la forme de Ix pour correspondre à Iy
        Iy = np.diff(Image, axis=0)[:, :-1] # Ajustement de la forme de Iy pour correspondre à Ix
        FM = np.mean(Ix ** 2 + Iy ** 2)

    elif algo == 'GRAE':  # Energy of gradient (Subbarao92a)
        Ix = np.copy(Image)
        Iy = np.copy(Image)
        # Calcul des différences entre les lignes et colonnes adjacentes de l'image
        Iy[:-1, :] = np.diff(Image, axis=0)
        Ix[:, :-1] = np.diff(Image, axis=1)
        # Calcul du carré des gradients
        FM = Ix**2 + Iy**2
        # Calcul de la moyenne des éléments de FM
        FM = np.mean(FM)

    elif algo == 'GRAS':  # Squared gradient (Eskicioglu95)
        Ix = np.diff(Image, axis=1)
        FM = np.mean(Ix ** 2)

    elif algo == 'GRAT':  # Thresholded gradient (Santos97)
        Th = 0  # Seuil (Threshold)
        Ix = np.copy(Image)
        Iy = np.copy(Image)
        # Calcul des différences entre les lignes et colonnes adjacentes de l'image
        Iy[:-1, :] = np.diff(Image, axis=0)
        Ix[:, :-1] = np.diff(Image, axis=1)
        # Calcul de la valeur maximale absolue entre les gradients dans les directions x et y
        FM = np.maximum(np.abs(Ix), np.abs(Iy))
        # Application du seuil
        FM[FM < Th] = 0
        # Calcul du pourcentage moyen des valeurs non nulles de FM
        FM = np.sum(FM[FM != 0]) / np.sum(FM != 0)

    elif algo == 'HELM':  # Helmli's mean method (Helmli2001)
        MEANF = np.ones((WSize, WSize)) / (WSize ** 2)
        U = convolve(Image, MEANF, mode='nearest')
#        R1 = np.divide(U, Image)
        R1 = np.divide(U, Image, out=np.zeros_like(U), where=Image!=0)
        R1[Image == 0] = 1
        index = (U > Image)
        FM = 1.0 / R1
        FM[index] = R1[index]
        FM = np.mean(FM)

#  See : algos.focus.entropy
#    elif algo == 'HISE':  # Histogram entropy (Krotkov86)
#        # Calculer l'entropie de l'image
#        FM = shannon_entropy(Image)

    elif algo == 'HISR':  # Histogram range (Firestone91)
        FM = np.max(Image) - np.min(Image)

    elif algo == 'LAPD':  # Diagonal Laplacian (Thelen2009)
        M1 = np.array([-1, 2, -1])
        M2 = np.array([[0, 0, -1], [0, 2, 0], [-1, 0, 0]]) / np.sqrt(2)
        M3 = np.array([[-1, 0, 0], [0, 2, 0], [0, 0, -1]]) / np.sqrt(2)
        F1 = convolve(Image, M1[np.newaxis, :], mode='nearest')
        F2 = convolve(Image, M2, mode='nearest')
        F3 = convolve(Image, M3, mode='nearest')
        F4 = convolve(Image, M1[np.newaxis, :].T, mode='nearest')
        FM = np.mean(np.abs(F1) + np.abs(F2) + np.abs(F3) + np.abs(F4))

    elif algo == 'LAPE':  # Energy of Laplacian (Subbarao92a)
        # Définition du filtre Laplacien
        LAP = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        )
        LAP_MATLAB = np.array(
            [[0.1667, 0.6667, 0.1667],
             [0.6667, -3.3333, 0.6667],
             [0.1667, 0.6667, 0.1667]]
        )
        # Appliquer le filtre laplacien à l'image
        FM = convolve(Image, LAP_MATLAB, mode='nearest') # type: ignore
        # Calculer la moyenne des valeurs au carré
        FM = np.mean(FM**2) # type: ignore

    elif algo == 'LAPM':  # Modified Laplacian (Nayar89)
        M = np.array([-1, 2, -1])
        Lx = convolve(Image, M[np.newaxis, :], mode='nearest')
        Ly = convolve(Image, M[np.newaxis, :].T, mode='nearest')
        FM = np.mean(np.abs(Lx) + np.abs(Ly))

    elif algo == 'LAPV':  # Variance of Laplacian (Pech2000)
        # Définition du filtre Laplacien
        LAP = np.array(
            [[0, 1, 0],
             [1, -4, 1],
             [0, 1, 0]]
        )
        LAP_MATLAB = np.array(
            [[0.1667, 0.6667, 0.1667],
             [0.6667, -3.3333, 0.6667],
             [0.1667, 0.6667, 0.1667]]
        )
        # Appliquer le filtre laplacien à l'image
        FM = convolve(Image, LAP_MATLAB, mode='nearest') # type: ignore
        # Calculer la variance des valeurs
        FM = np.var(FM) # type: ignore

    elif algo == 'SFIL':  # Steerable filters (Minhas2009)
        N = WSize // 2
        sig = N / 2.5

        # Création des filtres gaussiens et dérivés
        x, y = np.meshgrid(np.arange(-N, N+1), np.arange(-N, N+1))
        G = np.exp(-(x**2 + y**2) / (2 * sig**2)) / (2 * np.pi * sig)
        Gx = -x * G / (sig**2)
        Gy = -y * G / (sig**2)
        Gx /= np.sum(Gx)
        Gy /= np.sum(Gy)

        # Application des filtres pour détecter les gradients dans les directions x et y
        # Angles = [0 45 90 135 180 225 270 315];
        R = np.zeros((Image.shape[0], Image.shape[1], 8))
        R[:, :, 0] = convolve(Image, Gx, mode='nearest')
        R[:, :, 1] = convolve(Image, Gy, mode='nearest')
        for i in range(2, 8):
            theta = i * 45
            R[:, :, i] = np.cos(np.deg2rad(theta)) * R[:, :, 0] + np.sin(np.deg2rad(theta)) * R[:, :, 1]

        # Calcul de la mesure caractéristique
        FM = np.max(R, axis=2)
        FM = np.mean(FM)

    elif algo == 'SFRQ':  # Spatial frequency (Eskicioglu95)
        Ix = np.zeros_like(Image)
        Iy = np.zeros_like(Image)
        # Calcul du gradient dans la direction x
        Ix[:, :-1] = np.diff(Image, axis=1)
        # Calcul du gradient dans la direction y
        Iy[:-1, :] = np.diff(Image, axis=0)
        # Calcul de la magnitude du gradient
        FM = np.mean(np.sqrt(Ix**2 + Iy**2))

    elif algo == 'TENG':  # Tenengrad (Krotkov86)
        # Définition des filtres de Sobel
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
#         Sy = np.array([[-1, -2, -1],
#                        [0, 0, 0],
#                        [1, 2, 1]])
        # Calcul des gradients dans les directions x et y
        Gx = convolve(Image, Sx, mode='nearest')
        Gy = convolve(Image, Sx.T, mode='nearest')
        # Calcul de la magnitude du gradient
        FM = np.mean(Gx**2 + Gy**2) # type: ignore

    elif algo == 'TENV':  # Tenengrad variance (Pech2000)
        # Définition des filtres de Sobel
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
#         Sy = np.array([[-1, -2, -1],
#                        [0, 0, 0],
#                        [1, 2, 1]])
        # Calcul des gradients dans les directions x et y
        Gx = convolve(Image, Sx, mode='nearest')
        Gy = convolve(Image, Sx.T, mode='nearest')
        # Calcul de la magnitude du gradient
        FM = np.var(Gx**2 + Gy**2) # type: ignore

    elif algo == 'VOLA':  # Vollath's correlation (Santos97)
        # Calcul des différences entre les lignes de l'image
        I1 = np.copy(Image)
        I1[:-1, :] = Image[1:, :]
        I2 = np.copy(Image)
        I2[:-2, :] = Image[2:, :]
        Image = Image * (I1 - I2)
        FM = np.mean(Image)

    elif algo == 'WAVS':  # Sum of Wavelet coeffs (Yang2003)
        # Effectuer la décomposition en ondelettes avec un seul niveau
        coeffs = pywt.wavedec2(Image, 'db6', level=1)
        # Récupérer les coefficients de détail pour chaque composante
        cH, cV, cD = coeffs[1]
        # Calculer la mesure de caractéristique basée sur la somme des valeurs absolues
        FM = np.mean(np.abs(cH) + np.abs(cV) + np.abs(cD))

    elif algo == 'WAVR':  # Ratio of Wavelet coeffs (Yang2003)
        # Effectuer la décomposition en ondelettes avec trois niveaux
        coeffs = pywt.wavedec2(Image, 'db6', level=3)

        # Récupérer les coefficients de détail et d'approximation pour chaque niveau de décomposition
        cH, cV, cD = coeffs[1]
        cA1, cA2, cA3 = coeffs[0], coeffs[1], coeffs[2]

        # Calculer les valeurs absolues des coefficients de détail et d'approximation
        H = np.abs(cH)
        V = np.abs(cV)
        D = np.abs(cD)
        A1 = np.abs(cA1)
        A2 = np.abs(cA2)
        A3 = np.abs(cA3)

        # Calculer les sommes des carrés des coefficients de détail horizontaux, verticaux et diagonaux
        WH = np.mean(H**2 + V**2 + D**2)

        # Calculer les moyennes des valeurs absolues des coefficients d'approximation
        WL = np.mean(A1) + np.mean(A2) + np.mean(A3)

        # Calculer la mesure de caractéristique
        FM = WH / WL

    elif algo == 'WAVV':  # Variance of Wavelet coeffs (Yang2003)
        # Effectuer la décomposition en ondelettes avec un seul niveau
        coeffs = pywt.wavedec2(Image, 'db6', level=1)
        # Récupérer les coefficients de détail pour chaque composante
        cH, cV, cD = coeffs[1]
        # Calculer la mesure de caractéristique basée sur la somme des écarts types
        FM = np.std(cH) ** 2 + np.std(cV) ** 2 + np.std(cD) ** 2 # type: ignore

    else:
        raise ValueError(f"Unknown measure {Measure}")

    return FM


if __name__ == '__main__':
    from skimage import data
#    from skimage.io import imsave

    # Load image
    Image = data.camera()
    print(f"Image shape: {Image.shape}")
    print(f"Image type : {Image.dtype}")

    # Save image
    # imsave("camera.png", Image)

    for Measure in MEASURES_FOCUS:
        FM = fmeasure(Image, Measure)
        print(f"{Measure}: {np.round(FM, 3) if FM is not None and isinstance(FM, (int, float)) else None}")

#EOF