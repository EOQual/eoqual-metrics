#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exemple d'utilisation — évaluation par lot de la netteté d'un répertoire
d'images RGB TIFF. Script de démonstration, PAS un test automatisé (voir
docs/technique/limites.md §5.4).

Parcourt un répertoire, convertit chaque image en niveaux de gris, calcule
plusieurs métriques No-Reference de netteté reconnues comme fiables
(voir le choix justifié ci-dessous), et exporte un tableau récapitulatif
(pandas) en CSV — classé de l'image la plus nette à la plus floue.

Choix des métriques
--------------------
`eoqual.metrics.sharpness` expose 17 algos (voir docs/technique/no-reference.md
§3.3, docs/technique/bibliographie.md) — ce ne sont **pas** des implémentations
interchangeables d'une même définition (contrairement à `psnr(algo=...)`
par exemple) : chaque algo est une méthode de mesure différente. Plutôt
que de tout calculer (bruyant, redondant, certains algos très spécialisés
pour un usage satellite), ce script en retient sept, choisis pour couvrir
des principes complémentaires plutôt que des variantes d'un même principe :

- ``tenengrad_otsu``, ``laplacian`` — gradient, rapides, standards de
  longue date (Krotkov 1987 ; Pech-Pacheco et al. 2000).
- ``s3`` — spectral, combine pente du spectre de Fourier et variation
  totale locale (Vu & Chandler). Un bug de normalisation le rendant non
  monotone avec le flou a été corrigé (voir docs/technique/no-reference.md §3.3.c
  et la docstring du module) ; un léger résidu subsiste entre une image
  parfaitement nette et un flou d'environ 1 pixel, sans incidence sur le
  classement d'images plus franchement floues.
- ``cpbd``, ``blur_effect`` — perceptuels, validés par des tests
  subjectifs humains (Narvekar & Karam 2011 ; Crété et al. 2007) — les
  seuls de la sélection dont la corrélation au jugement humain a été
  mesurée expérimentalement par leurs auteurs.
- ``aem``, ``sasbem`` — famille *satellite*, les seules métriques du
  catalogue **physiquement calibrées** (MTF@Nyquist, méthode du bord
  incliné ISO 12233, sur arêtes naturelles de la scène) : comparables
  entre capteurs et dates d'acquisition, pas seulement entre images d'un
  même lot. Encore en développement — peuvent renvoyer un score peu
  fiable (`insufficient=True`) sur une scène sans arête exploitable
  (ciel uniforme, eau, nuages) ; ce script le détecte et le signale au
  lieu de mélanger un score fiable avec un score inexploitable.

Usage
-----
    PYTHONPATH=src python examples/batch_sharpness_assessment.py \\
        /chemin/vers/images --out resultats.csv
"""
__author__ = "Olivier Amram"
__status__ = "Development"

# ── Imports standard ─────────────────────────────────────────────────────
import sys
import os
import argparse
from pathlib import Path

# ── Imports tiers ────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.color import rgb2gray

# ── Imports eoqual.metrics ───────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import eoqual.metrics as m
from eoqual.metrics.nr.sharpness.backends.satellite.aem import aem
from eoqual.metrics.nr.sharpness.backends.satellite.sasbem import sasbem

# Algos "physiquement calibrés" (famille satellite) : appelés directement
# (pas via la façade m.sharpness, qui ne transmet pas return_details) pour
# distinguer un score fiable d'un score inexploitable (voir docstring du
# module ci-dessus).
_CALIBRATED_FUNCS = {"aem": aem, "sasbem": sasbem}

SHARPNESS_ALGOS = [
    "tenengrad_otsu",
    "laplacian",
    "s3",
    "cpbd",
    "blur_effect",
    "aem",
    "sasbem",
]


def find_tiff_images(directory: Path) -> list:
    """Liste les fichiers .tif/.tiff d'un répertoire (insensible à la casse), triés par nom."""
    files = {p for p in directory.iterdir() if p.suffix.lower() in (".tif", ".tiff")}
    return sorted(files)


def load_grayscale(path: Path) -> np.ndarray:
    """Charge une image RGB et la convertit en niveaux de gris, dynamique [0, 1]."""
    img = imread(str(path))
    if img.ndim == 3:
        # Ne garde que les 3 premiers canaux : ignore un éventuel canal alpha.
        gray = rgb2gray(img[..., :3])
    elif img.ndim == 2:
        img = img.astype(np.float64)
        rng = img.max() - img.min()
        gray = (img - img.min()) / rng if rng > 0 else np.zeros_like(img)
    else:
        raise ValueError(f"{path.name} : shape inattendue {img.shape}.")
    return gray.astype(np.float64)


def compute_row(path: Path) -> dict:
    """Calcule les 6 métriques de netteté retenues pour une image."""
    gray = load_grayscale(path)
    row = {"fichier": path.name}
    for algo in SHARPNESS_ALGOS:
        try:
            if algo in _CALIBRATED_FUNCS:
                score, details = _CALIBRATED_FUNCS[algo](gray, return_details=True)
                row[algo] = np.nan if details["insufficient"] else score
                row[f"{algo}_fiable"] = not details["insufficient"]
            else:
                row[algo] = m.sharpness(gray, algo=algo)
        except Exception as exc:  # une métrique indisponible ne doit pas bloquer les autres
            row[algo] = np.nan
            print(f"[avertissement] {path.name} / {algo} : {exc}", file=sys.stderr)
    return row


def assess_directory(directory: Path) -> pd.DataFrame:
    """Calcule les métriques de netteté sur chaque image TIFF du répertoire, triées de la plus nette à la plus floue."""
    files = find_tiff_images(directory)
    if not files:
        raise FileNotFoundError(f"Aucune image .tif/.tiff trouvée dans {directory}")

    df = pd.DataFrame([compute_row(path) for path in files]).set_index("fichier")

    # Rang moyen entre métriques (1 = la plus nette selon cette métrique) :
    # une façon simple de combiner des échelles non comparables entre elles
    # (chaque algo a sa propre unité/plage — voir docstring du module),
    # sans prétendre que les métriques se moyennent physiquement. Les
    # scores non fiables (aem/sasbem insuffisants) sont ignorés du rang
    # plutôt que traités comme "très flou".
    df["rang_moyen"] = df[SHARPNESS_ALGOS].rank(ascending=False).mean(axis=1)
    return df.sort_values("rang_moyen")


# ── Programme principal ──────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("directory", type=Path, help="Répertoire contenant les images .tif/.tiff")
    parser.add_argument("--out", type=Path, default=Path("sharpness_results.csv"),
                         help="Fichier CSV de sortie (défaut : sharpness_results.csv)")
    args = parser.parse_args()

    df = assess_directory(args.directory)

    pd.set_option("display.width", 160)
    print(df.to_string(float_format=lambda x: f"{x:.4f}"))

    df.to_csv(args.out)
    print(f"\nRésultats exportés : {args.out}")
    print(f"Image la plus nette (rang moyen toutes métriques) : {df.index[0]}")
