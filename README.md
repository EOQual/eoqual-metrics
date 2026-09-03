# eoqual-metrics

Collection de métriques de qualité image (Full-Reference, No-Reference),
pensée pour l'observation de la Terre — voir
**[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)** pour le détail de
chaque famille (méthode, implémentation, limites, évolutions envisagées)
et **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour l'audit
complet des licences du code tiers.

## Sommaire

1. [Installation](#installation)
2. [Ce qui est disponible selon l'installation](#ce-qui-est-disponible-selon-linstallation)
3. [Catalogue des métriques](#catalogue-des-métriques)
4. [Licence](#licence)
5. [Contribuer](#contribuer)
6. [Credits](#credits)

---

## Installation

### 1. Socle (obligatoire)

```bash
pip install git+https://github.com/EOQual/eoqual-metrics.git
```

Installe tout ce qui est nécessaire pour la majorité des métriques :
numpy, scipy, scikit-image, scikit-learn, opencv-python, sewar, pyrtools,
cpbd, pandas, rich, loguru... (liste complète : `pyproject.toml`,
section `[tool.poetry.dependencies]`).

**Prérequis système** — une seule bibliothèque C, requise par `pyfftw`
(lui-même nécessaire à l'étape 2 ci-dessous) :

```bash
brew install fftw      # macOS
conda install fftw     # conda
pixi add fftw          # pixi
```

### 2. `image-similarity-measures` (fortement recommandé)

Non installable automatiquement par poetry/pip (conflit de résolution de
dépendances avec `pyfftw` — voir le commentaire dans `pyproject.toml`) :
installation manuelle après l'étape 1, avec `--no-deps` pour réutiliser le
`pyfftw` déjà installé plutôt que d'en re-résoudre un autre :

```bash
pip install image-similarity-measures --no-deps
```

**Sans cette étape**, tout appel de `issm`, `sam`, `sre` (aucun autre
algorithme n'existe pour ces trois métriques) ou de `fsim` avec son algo
par défaut lève une `ImportError` explicite indiquant quoi installer —
voir [Ce qui est disponible selon l'installation](#ce-qui-est-disponible-selon-linstallation).
Pour `fsim` uniquement, une alternative existe sans cette étape :
`fsim(GT, P, algo="piq")` (nécessite l'extra `deep`, voir plus bas).

### 3. Extras optionnels

Avec le socle seul (étapes 1-2 ci-dessus, sans aucun extra), vous avez
déjà : toutes les métriques d'erreur, SNR/PSNR, `ssim`/`msssim` (algo par
défaut), corrélation, spectral, toute la famille `sharpness`, `niqe`,
`piqe`, `brisque` (algo par défaut — voir pourquoi ci-dessous), `entropy`,
`fmeasure`. Les deux extras suivants ne débloquent que des cas
spécifiques :

| Extra | Commande | Débloque | Poids |
|---|---|---|---|
| `svm` | `pip install "eoqual-metrics[svm]"` | Deux implémentations `brisque` **alternatives**, tierces (`image-quality`, `brisque`) — nécessite en plus `libsvm` (`conda install libsvm`) | léger |
| `deep` | `pip install "eoqual-metrics[deep]"` | `dists`, `vsi`, `lpips_vgg` (aucune alternative sans cet extra) ; algorithme `piq` en option supplémentaire pour `fsim`, `gmsd`, `vif`, `ssim`, `msssim` ; `nlpd` (via `IQA_pytorch`, seul backend) | ~2-3 Go (torch + torchvision) |

Backend par défaut de l'extra `deep` : **`piq`** (Apache-2.0). `IQA_pytorch`
(MIT, non maintenu depuis ~2021) reste disponible en algo `legacy`, et
c'est l'unique backend pour `nlpd`. `mad` reste **inutilisable quelle que
soit la dépendance installée** — voir `THIRD_PARTY_LICENSES.md`.

**Pourquoi `brisque` fonctionne-t-il sans l'extra `svm` ?** Son algo par
défaut (`numpy`) ne dépend d'aucune bibliothèque SVM externe : la
prédiction du modèle SVR pré-entraîné (noyau RBF) est recalculée
directement en numpy pur — lecture du fichier de modèle +
calcul du noyau à la main, sans passer par `libsvm` ni `scikit-learn` pour
l'inférence (voir `backends/brisque/numpy_impl.py`). L'extra `svm` ne
sert qu'à débloquer deux implémentations **tierces alternatives**
(`image-quality`, `brisque`), qui elles s'appuient sur de vraies
bibliothèques SVM — inutiles pour l'usage courant.

`cpbd` (dépendance de base, algo `sharpness(algo="cpbd")`) fonctionne dès
le socle sans action de votre part malgré son statut non maintenu — un
shim interne neutralise son incompatibilité avec SciPy récent.

---

## Ce qui est disponible selon l'installation

Récapitulatif : que se passe-t-il si une métrique est appelée sans que sa
dépendance soit installée ?

- **Algo par défaut sans dépendance disponible** → `ImportError` explicite
  à l'appel (précise quoi installer), ou `None`/score neutre selon la
  métrique (voir sa docstring).
- **Algo explicitement demandé** (`algo="xxx"`) **non enregistré pour
  cette bande** (GRAY/RGB) ou **inconnu** → `ValueError`.
- Aucune métrique ne plante à l'*import* du package : les dépendances
  optionnelles sont toutes chargées à la demande (`try/except ImportError`
  interne à chaque backend).

Pour lister, à tout moment, les algos réellement enregistrés pour chaque
métrique dans votre environnement :

```python
from eoqual.metrics.runner import list_metrics
list_metrics()  # tableau Rich : métrique, mode, algo par défaut, tous les algos
```

---

## Catalogue des métriques

### Full-Reference (FR) — comparer une image traitée à une référence

| Famille | Métriques |
|---|---|
| Structurelle (SSIM) | `ssim`, `msssim`, `cw_ssim` |
| Erreur pixel-à-pixel | `mae`, `mdae`, `mse`, `nrmse`, `rmse`, `rmse_sw` |
| Signal / bruit | `snr`, `psnr`, `psnrb`, `psnrc`, `wsnr` |
| Corrélation | `ncc`, `ndp`, `nmi`, `scc`, `uqi` |
| Perceptuelle | `dists`, `fsim`, `gmsd`, `issm`, `lpips_vgg`, `mad`, `nlpd`, `pamse`, `reco`, `vif`, `vsi` |
| Spectrale / télédétection | `ergas`, `rase`, `sam`, `sre` |

Détail de chaque métrique (principe, référence bibliographique) :
**[REFERENCE_TECHNIQUE.md §2](REFERENCE_TECHNIQUE.md)**.

### No-Reference (NR) — évaluer une image seule

- `brisque`, `niqe`, `piqe` — scores appris statistiquement, sans référence.
- `entropy`, `fmeasure` (27 opérateurs classiques de mise au point).
- `sharpness` — façade multi-algorithmes (`algo=`, voir
  `nr/sharpness/__init__.py`) :
    - **Gradient** : `tenengrad`, `laplacian`, `sobel`
    - **Phase** : `lpc_si`, `sharpness_index`, `psi`
    - **Spectral** : `s3`, `mtf` (MTF légère, arêtes ponctuelles)
    - **Perceptuel** (librairie externe) : `cpbd`
    - **Satellite** (calibré, remote sensing) : `brenner`, `fft`, `antonel`,
      `blur_kernel`, `wavelet`, `aem`, `sasbem` — `aem`/`sasbem` : MTF@Nyquist
      par slanted-edge method (ISO 12233) sur arêtes naturelles
      auto-détectées, sans mire de calibration ; `sasbem` ajoute un filtrage
      statistique robuste (MAD) et un indice de fiabilité.
- **Calcul par tuiles** (opt-in, toute métrique NR) :
  `eoqual.metrics.compute_score_on_tiles(...)` — voir
  `REFERENCE_TECHNIQUE.md §3.4` et `examples/tiled_vs_full_image.py`.

Détail : **[REFERENCE_TECHNIQUE.md §3](REFERENCE_TECHNIQUE.md)**.

### Reduced-Reference (RR)

Aucune à ce jour.

---

## Licence

**Le code propre à eoqual-metrics est sous licence MIT** (`license.txt`).

**Tous les algorithmes par défaut** (ceux obtenus sans
préciser `algo=`) sont sous licence permissive (MIT/BSD-3/Apache-2.0) ou
licence de recherche permissive avec citation (LIVE Lab, UT Austin — voir
`backends/niqe/models/NOTICE.md` et `backends/brisque/models/NOTICE.md`).

Seuls des **algorithmes optionnels** (jamais obtenus sans les demander
explicitement via `algo=`) restent sous une licence différente :

| Métrique | Algo concerné | Licence réelle |
|---|---|---|
| `wsnr` | `algo="metrikz"` (legacy — défaut : `numpy`) | GPLv2 |
| `mse`, `rmse`, `snr`, `psnr`, `psnrc`, `ssim`, `uqi`, `vif` | `algo="metrikz"` (un algo parmi plusieurs) | GPLv2 |
| `ssim` | `algo="ssim_3"` | GPL-3.0 |
| `ssim`, `msssim` | `algo="ssim_1"` / `"msssim_1"` | GPL-3.0/LGPL-3.0 |
| `ssim` | `algo="ssim_2"` | GPL (auto-déclarée) |
| `ssim` | `algo="ssim_4"` | GPL (auto-déclarée) |

Aucun de ces algos n'est atteint par un appel « par défaut » de la
librairie. Inventaire complet, fichier par fichier, avec vérification de
chaque source : **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)**.

---

## Contribuer

Bug, métrique manquante, algorithme à ajouter, erreur de licence
constatée : les contributions sont bienvenues — voir
**[CONTRIBUTING.md](CONTRIBUTING.md)** pour comment signaler un problème,
proposer une métrique, ou soumettre une implémentation.

---

## Credits

- cpbd [https://pypi.org/project/cpbd/]
- image-similarity-measures [https://pypi.org/project/image-similarity-measures/]
- IQA-optimization [https://github.com/dingkeyan93/IQA-optimization]
- IQA_pytorch [https://pypi.org/project/IQA-pytorch/] (legacy, voir THIRD_PARTY_LICENSES.md)
- piq [https://github.com/photosynthesis-team/piq] (backend par défaut de dists/fsim/gmsd/vif/ssim/msssim/vsi/lpips_vgg)
- metrikz [https://gitlab.com/gpds-unb/pymetrikz]
- NumPy [https://numpy.org/]
- OpenCV [https://opencv.org/]
- pyrtools [https://pypi.org/project/pyrtools/]
- Scikit-Image [https://scikit-image.org/]
- Scikit-Learn [https://scikit-learn.org/stable/]
- SciPy [https://scipy.org/]
- sewar [https://sewar.readthedocs.io/en/latest/]
- fmeasure [https://fr.mathworks.com/matlabcentral/fileexchange/27314-focus-measure]

- BRISQUE
    - [https://github.com/ocampor/image-quality]
    - [https://github.com/EadCat/NIQA/tree/master/brisque]
    - [https://github.com/buyizhiyou/NRVQA]
    - [https://github.com/rehanguha/brisque]
- CW_SSIM
    - [https://github.com/charparr/tundra-snow/blob/master/continuous_patterns.py]
- MSSSIM
    - [https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py]
- NIQE
    - [https://programtalk.com/vs2/?source=python%2F341%2Fvideo-quality#]
    - [https://github.com/EadCat/NIQA/tree/master/niqe]
- PIQE
    - [https://github.com/EadCat/NIQA/tree/master/piqe]
- RECO
    - [https://programtalk.com/vs2/?source=python%2F341%2Fvideo-quality#]
- SSIM
    - [https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py]
    - [https://github.com/aizvorski/video-quality]
    - [http://isit.u-clermont1.fr/~anvacava/code-ssim-source.html]
- VIF
    - [https://programtalk.com/vs2/?source=python%2F341%2Fvideo-quality#]
    - [https://github.com/abhinaukumar/vif/blob/main/vif_utils.py]
