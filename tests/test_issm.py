"""ISSM (Aljanabi et al., 2019), algo 'numpy' : propriétés attendues d'après l'article."""
import numpy as np
import pytest
from scipy import ndimage
from skimage import data

from eoqual.metrics import issm


@pytest.fixture(scope="module")
def images():
    ref = data.camera().astype(np.float64)
    coins = data.coins().astype(np.float64)
    other = ndimage.zoom(coins, np.array(ref.shape) / np.array(coins.shape), order=1)
    return ref, other


def noisy(ref, psnr_db, seed=0):
    sigma = 255.0 / 10 ** (psnr_db / 20)
    return ref + np.random.default_rng(seed).normal(0, sigma, ref.shape)


def test_identical_images_give_one(images):
    ref, _ = images
    assert issm(ref, ref) == pytest.approx(1.0)


def test_decreases_with_noise_like_figures_3_to_6(images):
    ref, _ = images
    values = [issm(ref, noisy(ref, psnr)) for psnr in (60, 30, 20, 10, 0)]
    assert values[0] > 0.95
    assert all(a > b for a, b in zip(values, values[1:]))


def test_unrelated_images_near_zero_like_figures_7_to_10(images):
    ref, other = images
    for psnr in (60, 20, 10):
        assert issm(noisy(ref, psnr), other) < 0.05


def test_independent_of_value_scale(images):
    ref, _ = images
    p = noisy(ref, 20)
    assert issm(ref / 255.0, p / 255.0) == pytest.approx(issm(ref, p), rel=1e-9)
    assert issm(ref * 16, p * 16, data_range=255 * 16) == pytest.approx(issm(ref, p, data_range=255), rel=1e-9)


def test_probabilities_variant_follows_equations(images):
    # Équations (10)-(13) à la lettre : < 1 pour des images identiques et
    # nettement > 0 pour des images sans rapport (contredit les figures).
    ref, other = images
    assert 0.85 < issm(ref, ref, ehs="probabilities") < 1.0
    assert issm(ref, other, ehs="probabilities") > 0.15


def test_uniform_and_errors(images):
    # En comptes, EHS < 0 : pour deux images identiques I = 1 + 0.7 / (0.8 |EHS|),
    # léger dépassement visible seulement sur une petite image.
    flat = np.full((32, 32), 7.0)
    assert issm(flat, flat) == pytest.approx(1.0, rel=1e-3)
    ref, _ = images
    with pytest.raises(ValueError):
        issm(np.stack([ref] * 3, axis=-1), np.stack([ref] * 3, axis=-1))
    with pytest.raises(ValueError):
        issm(ref, ref, ehs="autre")
    with pytest.raises(ValueError):
        issm(ref, ref, algo="inconnu")
