import numpy as np
import numpy.typing as npt
import cv2
from skimage.measure import shannon_entropy
from scipy.stats import entropy as entropy_scipy


__all__ = [
    "entropy",
]

MEASURES_ENTROPY = [
    "skimage",
    "entropy_1",
    "scipy",
]

def _entropy_1(P: npt.NDArray) -> float:
    """
    Entropy of gray image
    """
    if len(P.shape) == 3:
        image = cv2.cvtColor(P, cv2.COLOR_BGR2GRAY) # type: ignore

    if type(P).__module__ != np.__name__:
        image = np.asarray(P).flatten()
    else:
        image = P.flatten()
    
    image = image + 0.5
    image = image.astype(np.int32)

    n = len(image)

    if image is None or n <= 1:
        return 0

    counts = np.bincount(image)
    counts = counts[counts > 0]
    if len(counts) == 1:
        return 0

    probs = counts.astype(float) / n

    return -np.sum(probs * np.log2(probs))  # type: ignore


def entropy(P: npt.NDArray, algo: str='entropy_1') -> float: # type: ignore
    """
    Computes the entropy metric, no-reference image quality score
   
    Parameters
    ----------
    P : npt.NDArray
        input image
    algo: str
        1. entropy_1

    Returns
    -------
    float
        entropy value
    """

    if algo == 'skimage':  # Histogram entropy (Krotkov86)
        # Calculer l'entropie de l'image
        return(shannon_entropy(P))
    elif algo == 'entropy_1':
        return _entropy_1(P)
    elif algo == 'scipy':
        hist, _ = np.histogram(P.flatten(), bins=256, density=True)
        return(entropy_scipy(hist, base=2)) # type: ignore
    else:
        raise ValueError(f"'algo' value [{algo}] is not correct")

#EOF