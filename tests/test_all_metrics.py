#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test toutes les métriques sur une image synthétique.

Usage
-----
    PYTHONPATH=src python tests/test_all_metrics.py

Remplace l'ancien test_all_zMetrics.py qui utilisait ObjMetrics / zMetrics.
"""
__author__    = "Olivier Amram"
__version__   = "20260509"
__status__    = "Development"

# ── Imports standard ──────────────────────────────────────────────────────────
import sys
import os
from time import time

# ── Imports tiers ─────────────────────────────────────────────────────────────
from loguru import logger
import numpy as np
from skimage import data  # type: ignore

# ── Imports eoqual.metrics ──────────────────────────────────────────────────────
# Ajoute src/ au path pour exécution sans installation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import eoqual.metrics as m
from eoqual.metrics.runner import MetricsRunner, list_metrics
from eoqual.metrics.export.console import display_table
from eoqual.metrics.export.csv import save_csv
from eoqual.metrics.config import METRICS_CONFIGS


# ── Helpers ───────────────────────────────────────────────────────────────────
def add_gaussian_noise(image: np.ndarray, mean: float = 0, sigma: float = 20) -> np.ndarray:
    """
    Ajoute un bruit gaussien à une image ``uint8``.

    Parameters
    ----------
    image : np.ndarray
        Image d'entrée.
    mean : float
        Moyenne du bruit. Par défaut ``0``.
    sigma : float
        Écart-type du bruit. Par défaut ``20``.

    Returns
    -------
    np.ndarray
        Image bruitée clippée dans ``[0, 255]`` (dtype ``uint8``).
    """
    noise = np.random.normal(mean, sigma, image.shape)
    noisy = np.clip(image + noise, 0, 255).astype(np.uint8)
    return noisy


# ── Programme principal ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # Logger
    logger.remove()
    fmt = "{time} | {level: <8} | {name: ^15} | {function: ^17} | {line: >3} | {message}"
    logger.add(sys.stdout, format=fmt, level="DEBUG")
    logger.add(lambda _: sys.exit(1), level="ERROR")

    # ── Vérification cohérence __all__ / METRICS_CONFIGS ─────────────────────
    # Les noms de métriques dans __all__ qui sont aussi des clés de METRICS_CONFIGS
    # doivent apparaître dans le même ordre.
    metrics_in_all = [name for name in m.__all__ if name in METRICS_CONFIGS]
    metrics_in_cfg = list(METRICS_CONFIGS.keys())
    if metrics_in_all != metrics_in_cfg:
        logger.error(
            f"Désynchronisation __all__ / METRICS_CONFIGS !\n"
            f"  __all__ metrics : {metrics_in_all}\n"
            f"  METRICS_CONFIGS : {metrics_in_cfg}"
        )
        sys.exit(1)
    logger.info("✓ __all__ et METRICS_CONFIGS sont synchronisés")

    # ── Chargement images ─────────────────────────────────────────────────────
    Image       = data.camera()       # (512, 512) uint8
#    Image       = data.astronaut()    # (512, 512, 3) uint8
    bands       = 'GRAY' if Image.ndim == 2 else 'RGB'
    Image_noisy = add_gaussian_noise(Image)
    logger.info(f"Image shape : {Image.shape}  dtype : {Image.dtype}")

    list_metrics()


    # ── Sélection des métriques ───────────────────────────────────────────────
    # Dictionnaire vide = toutes les métriques par défaut
    # Exemple sélection manuelle :
#    dictMetrics = {'psnr': 'numpy', 'ssim': 'skimage', 'brisque': 'libsvm_impl'}
    dictMetrics = {
        'psnr': 'numpy'
    }

    t0 = time()

    # ── Runner ────────────────────────────────────────────────────────────────
    runner = MetricsRunner(
        bands=bands,     # 'GRAY' ou 'RGB' (si applicable)
        metrics=dictMetrics,
        exclude=[
        ],          # ex. ['niqe', 'piqe'] pour exclure les métriques lentes
        default_only=False,
    )

    if runner.is_empty:
        logger.warning("Aucune métrique planifiée !")
        sys.exit(0)

    # Calcul sur une paire
    runner.compute(Image, Image_noisy)
    runner.accumulate()

    logger.info(f"Durée totale : {time() - t0:.2f}s")

    # ── Affichage ─────────────────────────────────────────────────────────────
    display_table(
        runner.results_accumulated,
        METRICS_CONFIGS,
        orientation="horizontal",
        legend="Métriques de Performance",
        output="console",      # "html" ou "both" pour ouvrir dans le navigateur
    )

    # ── Export CSV ────────────────────────────────────────────────────────────
    save_csv(runner.results_accumulated, "outputMetrics.csv")

# EOF