#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exemple d'utilisation — comparaison calcul pleine image vs. calcul par
tuiles, pour la même métrique. Script de démonstration, PAS un test
automatisé (voir REFERENCE_TECHNIQUE.md §5.4).

Quand préférer l'un ou l'autre
------------------------------
- **Pleine image** (`m.sharpness(image, algo=...)`) : un seul nombre,
  rapide, suffisant quand la scène est homogène ou que seule une tendance
  globale importe.
- **Par tuiles** (`m.compute_score_on_tiles(...)`) : une carte de scores,
  utile sur une scène large et hétérogène (typiquement satellite) où un
  score unique masquerait une dégradation localisée (flou de bougé sur
  une portion de l'image, zone nuageuse, etc.) — voir
  `eoqual.metrics.tiling` pour la méthodologie complète (indice de
  représentativité par tuile, log résumé).

Usage
-----
    PYTHONPATH=src python examples/tiled_vs_full_image.py
"""
import os
import sys
from functools import partial

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import eoqual.metrics as m
from eoqual.metrics.nr.sharpness.backends.satellite.aem import aem


if __name__ == "__main__":
    image_path = os.path.join(os.path.dirname(__file__), '..', 'tests', 'images', 'r0.png')
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype(np.float32)
    print(f"Image : {image_path}  shape={image.shape}")

    # ── 1. Pleine image ──────────────────────────────────────────────────
    # Un score unique, moyenné sur toute la scène.
    score_full = m.sharpness(image, algo="tenengrad")
    print(f"\nTenengrad (pleine image) : {score_full:.2f}")

    aem_full = aem(image)
    print(f"AEM (pleine image)       : {aem_full:.4f}")

    # ── 2. Par tuiles ─────────────────────────────────────────────────────
    # Même métrique, mais une ligne par tuile — révèle l'hétérogénéité
    # spatiale que le score pleine image masquait.

    # 2a. Métrique sans détail natif (tenengrad) : représentativité =
    #     variance de la tuile (comportement par défaut).
    df_tenengrad = m.compute_score_on_tiles(
        image, partial(m.sharpness, algo="tenengrad"), tile_size=128, overlap=0.0,
    )
    print(f"\nTenengrad par tuiles ({len(df_tenengrad)} tuiles) :")
    print(f"  min={df_tenengrad['score'].min():.2f}  "
          f"médiane={df_tenengrad['score'].median():.2f}  "
          f"max={df_tenengrad['score'].max():.2f}")
    print("  -> un score pleine image unique (ci-dessus) ne montre pas cette dispersion.")

    # 2b. Métrique avec détail natif (aem) : return_details=True active
    #     l'indice de fiabilité propre à la métrique (n_edges ici), et
    #     signale les tuiles sans arête exploitable (insufficient=True)
    #     plutôt que de les mélanger silencieusement aux scores valides.
    df_aem = m.compute_score_on_tiles(
        image, aem, tile_size=128, overlap=0.0,
        method_kwargs={"return_details": True}, metric_name="aem",
    )
    n_insufficient = int(df_aem["insufficient"].sum())
    print(f"\nAEM par tuiles ({len(df_aem)} tuiles, "
          f"{n_insufficient} insuffisante(s)) :")
    reliable = df_aem[~df_aem["insufficient"]]
    if len(reliable):
        print(f"  médiane (tuiles fiables uniquement) = {reliable['score'].median():.4f}")
    print(df_aem[["row", "col", "score", "n_edges", "insufficient"]].to_string(index=False))
