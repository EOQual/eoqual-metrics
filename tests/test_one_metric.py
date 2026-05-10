#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calcule une métrique qualité sur une image (vs. sa référence).

Usage
-----
    PYTHONPATH=src python tests/test_one_metric.py \\
        --P tests/images/r1.png --GT tests/images/r0.png --metric psnr

    # Métrique NR (sans GT) :
    PYTHONPATH=src python tests/test_one_metric.py \\
        --P tests/images/r1.png --metric sharpness --algo tenengrad

Remplace l'ancien test_one_zMetrics.py qui importait zMetrics directement.
"""
__author__  = "Olivier Amram"
__version__ = "20260509"
__status__  = "Development"

# ── Imports standard ──────────────────────────────────────────────────────────
import sys
import os
import argparse

# ── Imports tiers ─────────────────────────────────────────────────────────────
import cv2
import matplotlib.pyplot as plt

# ── Imports eoqual.metrics ──────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import eoqual.metrics as m
from eoqual.metrics.config import METRICS_CONFIGS


# ── Programme principal ───────────────────────────────────────────────────────
if __name__ == "__main__":

    # Liste des métriques disponibles : toutes les clés de METRICS_CONFIGS
    available_metrics = list(METRICS_CONFIGS.keys())

    parser = argparse.ArgumentParser(description="Calcul d'une métrique sur une image")
    parser.add_argument(
        '--metric',
        choices=available_metrics,
        required=True,
        help="Nom de la métrique à calculer",
    )
    parser.add_argument(
        '--algo',
        default=None,
        help="Algorithme à utiliser (défaut : algo par défaut de la métrique)",
    )
    parser.add_argument('--P',  required=True,      help="Image à qualifier (chemin)")
    parser.add_argument('--GT', default='',          help="Image de référence (chemin, FR uniquement)")
    parser.add_argument('--no-plot', action='store_true', help="Désactiver l'affichage")
    args = parser.parse_args()

    metric_name: str = args.metric
    metric_cfg         = METRICS_CONFIGS[metric_name]
    algo_name: str     = args.algo or metric_cfg['default']

    # Vérification que l'algo existe pour cette métrique
    valid_algos = [a['name'] for a in metric_cfg['algo']]
    if algo_name not in valid_algos:
        print(f"Erreur : algo '{algo_name}' invalide pour '{metric_name}'.")
        print(f"Algos disponibles : {valid_algos}")
        sys.exit(1)

    # ── Chargement des images ─────────────────────────────────────────────────
    P_img = cv2.imread(args.P)
    if P_img is None:
        raise FileNotFoundError(f"Image introuvable : {args.P}")
    if len(P_img.shape) > 2:
        P_img = cv2.cvtColor(P_img, cv2.COLOR_BGR2GRAY)
    print(f"P shape : {P_img.shape}  dtype : {P_img.dtype}")

    GT_img = None
    if args.GT:
        GT_img = cv2.imread(args.GT)
        if GT_img is None:
            raise FileNotFoundError(f"Image GT introuvable : {args.GT}")
        if len(GT_img.shape) > 2:
            GT_img = cv2.cvtColor(GT_img, cv2.COLOR_BGR2GRAY)

    # ── Calcul ────────────────────────────────────────────────────────────────
    fn = getattr(m, metric_name)

    # Déterminer la signature (FR ou NR, map ou non)
    algo_def = next(a for a in metric_cfg['algo'] if a['name'] == algo_name)
    has_map  = 'map' in algo_def
    is_fr    = metric_cfg['metric_mode'] == 'FR'

    if is_fr:
        if GT_img is None:
            print(f"Erreur : '{metric_name}' est une métrique FR — --GT est requis.")
            sys.exit(1)
        result = fn(GT_img, P_img, algo=algo_name)
    else:
        result = fn(P_img, algo=algo_name)

    if has_map and isinstance(result, tuple):
        score, score_map = result
    else:
        score = result
        score_map = None

    print(f"\n{metric_name} [{algo_name}] : {score}\n")

    # ── Affichage ─────────────────────────────────────────────────────────────
    if not args.no_plot:
        n_axes = 2 if GT_img is not None else 1
        if score_map is not None:
            n_axes += 1
        fig, axes = plt.subplots(1, n_axes, figsize=(5 * n_axes, 5))
        if n_axes == 1:
            axes = [axes]

        ax_idx = 0
        if GT_img is not None:
            axes[ax_idx].imshow(GT_img, cmap='gray')
            axes[ax_idx].set_title('GT (référence)')
            ax_idx += 1

        axes[ax_idx].imshow(P_img, cmap='gray')
        axes[ax_idx].set_title(f'P  —  {metric_name}={score:.4f}' if score is not None else 'P')
        ax_idx += 1

        if score_map is not None:
            axes[ax_idx].imshow(score_map, cmap='hot')
            axes[ax_idx].set_title(f'Carte {metric_name}')

        plt.tight_layout()
        plt.show()

# EOF
