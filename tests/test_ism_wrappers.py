"""Enveloppes image-similarity-measures : forme des images et sens de SAM.

Sautés si ``image-similarity-measures`` n'est pas installé (installation
manuelle, voir README).
"""
import numpy as np
import pytest

from eoqual.metrics.core.checker import as_channels_last

qm = pytest.importorskip("image_similarity_measures.quality_metrics")
import eoqual.metrics as m  # noqa: E402

ISM = "image-similarity-measures"
# nom dans eoqual.metrics -> nom dans image-similarity-measures. Comparaison
# directe seulement pour les fonctions sans hypothèse de dynamique ; psnr,
# rmse, ssim, fsim et issm sont testées pour leur indépendance à l'échelle.
DIRECT = {"uqi": "uiq", "sre": "sre", "sam": "sam"}
SCALE_DEPENDENT = ("psnr", "rmse", "ssim", "fsim", "issm")


@pytest.fixture(scope="module")
def images():
    rng = np.random.default_rng(0)
    rgb = (rng.random((64, 64, 3)) * 200 + 20).astype(np.float32)
    noisy = np.clip(rgb + rng.normal(0, 10, rgb.shape), 0, 255).astype(np.float32)
    return rgb, noisy


def test_as_channels_last():
    assert as_channels_last(np.zeros((4, 5))).shape == (4, 5, 1)
    assert as_channels_last(np.zeros((4, 5, 3))).shape == (4, 5, 3)
    with pytest.raises(ValueError):
        as_channels_last(np.zeros((2, 4, 5, 3)))


@pytest.mark.parametrize("name", sorted(DIRECT))
def test_rgb_wrapper_matches_direct_call(images, name):
    gt, p = images
    expected = float(getattr(qm, DIRECT[name])(gt, p))
    assert getattr(m, name)(gt, p, ISM) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("name", sorted(set(DIRECT) - {"sam"}))
def test_gray_wrapper_matches_direct_call(images, name):
    gt, p = (x.mean(axis=2) for x in images)
    expected = float(getattr(qm, DIRECT[name])(gt[..., None], p[..., None]))
    assert getattr(m, name)(gt, p, ISM) == pytest.approx(expected, rel=1e-6)


def test_sam_is_spectral_angle_in_degrees(images):
    gt, p = (x.astype(np.float64) for x in images)
    reference = np.degrees(np.mean(np.arctan2(
        np.linalg.norm(np.cross(gt, p), axis=-1), np.sum(gt * p, axis=-1))))
    assert m.sam(*images, ISM) == pytest.approx(reference, rel=1e-4)
    assert m.sam(images[0], images[0], ISM) == pytest.approx(0.0, abs=1e-3)


def test_sam_rejects_single_channel(images):
    with pytest.raises(ValueError):
        m.sam(images[0].mean(axis=2), images[1].mean(axis=2), ISM)


@pytest.mark.parametrize("name", SCALE_DEPENDENT)
def test_independent_of_value_scale(images, name):
    gt, p = (x.mean(axis=2) for x in images)
    reference = getattr(m, name)(gt, p, ISM)
    for scale in (1 / 255, 16.0):
        assert getattr(m, name)(gt * scale, p * scale, ISM) == pytest.approx(reference, rel=1e-6)


def test_psnr_ssim_match_default_algorithms(images):
    gt, p = (x.mean(axis=2) for x in images)
    assert m.psnr(gt, p, ISM) == pytest.approx(m.psnr(gt, p, "numpy"), rel=1e-6)  # ISM en float32
    assert m.ssim(gt, p, ISM) == pytest.approx(m.ssim(gt, p, "skimage"), rel=1e-6)


def test_issm_ism_identical_images(images):
    gt = images[0].mean(axis=2)
    assert m.issm(gt, gt, ISM) == pytest.approx(1.0, rel=1e-3)  # EHS en comptes : 1 + 0.7/(0.8|EHS|)
    assert m.issm(gt, gt + np.random.default_rng(1).normal(0, 20, gt.shape), ISM) < 1.0
