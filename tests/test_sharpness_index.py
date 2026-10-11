"""Tests de l'indice de netteté SI."""
import numpy as np
import pytest
from scipy import ndimage

from eoqual.metrics.nr.sharpness.backends.phase.sharpness_index import (
    _perdecomp,
    sharpness_index,
    sharpness_stats,
)


def _smooth_scene(shape=(64, 80), seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    u = ndimage.gaussian_filter(rng.standard_normal(shape), 2)
    return u + (((x - 40) ** 2 + (y - 30) ** 2) < 15 ** 2)


def test_sharpness_moments_match_monte_carlo():
    u, _ = _perdecomp(_smooth_scene())
    st = sharpness_stats(u, pmode=0)
    M, N = u.shape
    U = np.fft.fft2(u)
    rng = np.random.default_rng(1)

    def tv(v):
        return np.abs(np.diff(np.pad(v, ((0, 0), (0, 1)), mode="wrap"), axis=1)).sum() + \
               np.abs(np.diff(np.pad(v, ((0, 1), (0, 0)), mode="wrap"), axis=0)).sum()

    samples = [tv(np.real(np.fft.ifft2(U * np.fft.fft2(rng.standard_normal((M, N)) / np.sqrt(M * N)))))
               for _ in range(1500)]
    assert np.isclose(np.mean(samples), st["mu"], rtol=2e-3)
    assert np.isclose(np.std(samples), st["sigma"], rtol=0.06)


def test_sharpness_decreases_with_blur_and_noise():
    u = _smooth_scene((128, 128), seed=2) * 100
    # flou fort : le SI atteint un plancher (plus de contour cohérent), d'où sigma <= 2
    si = [sharpness_index(ndimage.gaussian_filter(u, s)) for s in (0, 0.5, 1, 2)]
    assert all(a > b for a, b in zip(si, si[1:]))
    rng = np.random.default_rng(3)
    assert sharpness_index(u + rng.normal(0, 20, u.shape)) < si[0]


def test_sharpness_constant_image():
    assert sharpness_index(np.full((16, 16), 7.0)) == 0.0


def test_perdecomp_properties():
    u = np.random.default_rng(0).random((37, 50)) + np.linspace(0, 3, 50)[None, :]
    p, s = _perdecomp(u)
    assert np.allclose(p + s, u) and abs(s.mean()) < 1e-12


@pytest.mark.parametrize("pmode", [0, 1, 2, 3])
def test_pmodes_and_errors(pmode):
    assert sharpness_index(_smooth_scene() * 100, pmode=pmode) > 0
    with pytest.raises(ValueError):
        sharpness_index(np.zeros((8, 8, 3)))
    with pytest.raises(ValueError):
        sharpness_index(np.zeros((8, 8)), pmode=4)
