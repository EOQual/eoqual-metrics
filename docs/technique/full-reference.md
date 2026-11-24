# Full-Reference : comparer deux images

## 2. Full-Reference (FR)

Comparent une image traitée `P` à une référence `GT` (même shape).

### 2.1 Erreur pixel-à-pixel

**Méthode** : écart brut entre pixels homologues — moyenne (`mae`, `mse`),
racine (`rmse`), médiane (`mdae`), version normalisée par la dynamique
(`nrmse`), ou calculée par fenêtre glissante (`rmse_sw`).

**Implémentation** : `fr/error.py`. `mae`/`mdae`/`mse`/`rmse` : `numpy`
seul algo. `nrmse` : `skimage`, seul algo. `rmse_sw` : `sewar`, seul algo.
D'autres bibliothèques (`skimage`/`metrikz`/`sewar`/
`image-similarity-measures`) ne sont pas proposées pour `mse`/`rmse` :
leur résultat est identique (résultat bit-exact, temps de calcul équivalent sur
image réelle et synthétique ; détail dans [formulaire](formulaire.md)), sauf
`image-similarity-measures` pour `rmse`, qui calculait en réalité une
*autre* métrique (RMSE normalisée par la dynamique, pas la RMSE brute —
écart ~99,6 %, cohérent avec une division par ~255). Appeler ces
fonctions avec l'un de ces noms d'algo reste accepté (compatibilité
ascendante) : un avertissement est émis et le calcul se fait avec
`numpy`.

**Limites** : mesures non perceptuelles (une MSE faible ne garantit pas une
image visuellement fidèle) — voir §2.3/2.5 pour des alternatives
perceptuelles. `rmse_sw` : l'algo `map=True` de sewar est désactivé dans
`config.py` (erreur de type constatée, jamais corrigée).

### 2.2 Signal-based (SNR/PSNR)

**Méthode** : rapport entre puissance du signal et puissance de l'erreur,
en dB. `psnrb` ajoute une pénalité de blocking (compression JPEG-like).
`psnrc` = variante couleur.

**Implémentation** : `fr/signal.py`. `psnr`/`psnrc` : `numpy` seul algo —
`skimage`/`metrikz`/`sewar`/`image-similarity-measures` non proposés
(résultat bit-exact, temps de calcul équivalent ; détail dans
[formulaire](formulaire.md)) ; ces noms d'algo restent acceptés (avertissement, calcul
fait avec `numpy`). `psnrb` : `sewar` seul algo. `snr`/`wsnr`
(pondération perceptuelle par CSF — Contrast Sensitivity Function de
Mannos & Sakrison) : défaut `numpy`, `metrikz` conservé en algo legacy —
vendored dans le projet (pas une dépendance externe), bit-exact et à
vitesse équivalente à `numpy` sur les mêmes bancs d'essai, mais pas
encore retiré (voir `THIRD_PARTY_LICENSES.md` pour son statut de licence).

**Limites** : PSNR en général : peu corrélé à la perception humaine
au-delà d'un ordre de grandeur grossier — préférer SSIM/VIF/DISTS pour une
évaluation perceptuelle.

### 2.3 Famille SSIM (similarité structurelle)

**Méthode** : Wang et al. 2004 — compare luminance, contraste et
structure dans des fenêtres locales glissantes. `msssim` = version
multi-échelle (pyramide). `cw_ssim` = variante en ondelettes complexes,
robuste aux petits décalages/rotations (translation-invariant).

**Implémentation** : `fr/ssim.py`. `ssim` défaut `skimage` (GRAY+RGB) ;
7 algos alternatifs dont 4 vendored (`ssim_1`..`ssim_4`, voir [§5.2](limites.md#52-licences-tierces) licences),
plus `piq` (Apache-2.0, GRAY+RGB) et `IQA_pytorch` (RGB,
legacy). `msssim` défaut `sewar`, alt. `piq` (≥161×161 px). `cw_ssim`
défaut `cw_ssim_1` (seul algo actif, GRAY uniquement, MIT — voir
`backends/cw_ssim/wavelet_impl.py`).

**Limites** : les algos `piq`/`IQA_pytorch` de `ssim`/`msssim` nécessitent
l'extra `deep` (torch). `cw_ssim` n'a pas d'implémentation RGB active
(celle d'IQA_pytorch, cassée, n'est pas au registre ; `piq` n'implémente
pas CW-SSIM). Pas de version par tuiles pour la famille SSIM (calcul
toujours pleine image — voir [§3.4](no-reference.md#34-calcul-par-tuiles-no-reference-opt-in) pour le tuilage NR).

### 2.4 Corrélation / statistique

**Méthode** : `ncc`/`ndp` (corrélation croisée normalisée), `nmi`
(information mutuelle normalisée — dépendance statistique, pas seulement
linéaire), `scc` (corrélation spatiale), `uqi` (Universal Quality Index —
combine luminance/contraste/corrélation, précurseur de SSIM), `jsd`
(distance de Jensen-Shannon — version symétrisée et bornée de la
divergence de Kullback-Leibler, entre histogrammes d'intensité plutôt
qu'entre pixels ; nombre de classes adaptatif par la règle de
Freedman-Diaconis, voir docstring). Seule métrique de cette famille qui
ignore toute correspondance spatiale pixel à pixel — utile pour repérer
un décalage radiométrique global (exposition, calibration capteur,
conditions atmosphériques) même sur des images mal recalées.

**Implémentation** : `fr/correlation.py`. `ncc`/`ndp`/`jsd` : `numpy` seul
algo — `ncc` acceptait aussi `scipy`, `ndp` acceptait `scipy_1`/`scipy_2`/
`sklearn` ne sont pas proposés (résultat quasi bit-exact, écart
< 2e-13 ; temps de calcul équivalent ; détail dans [formulaire](formulaire.md)) — ces
noms d'algo restent acceptés (avertissement, calcul fait avec
`numpy`). `scikit-learn` n'est pas une dépendance. `nmi` : seul algo `skimage`. `scc`/`uqi` :
`sewar` (+ alternatives pour `uqi`, voir [§5.2](limites.md#52-licences-tierces)).

**Limites** : peu utilisées en pratique face à SSIM/PSNR — surtout utiles
en recalage/imagerie multimodale (`nmi`, `ncc`). `jsd` : histogramme
1-D global, insensible à toute réorganisation spatiale des pixels (deux
images très différentes structurellement peuvent avoir des distributions
d'intensité proches) — à utiliser en complément, pas en remplacement,
des métriques structurelles/spatiales.

### 2.5 Perceptuelle (deep learning et modèles physiologiques)

**Méthode** : modèles calés sur la perception humaine — `vif` (fidélité
informationnelle, modèle de canal bruité), `fsim` (congruence de phase +
gradient), `gmsd` (déviation de similarité de magnitude de gradient),
`issm` (entropie de l'histogramme conjoint + SSIM + corrélation de
contours Canny — Aljanabi et al. 2019), `dists`/`lpips_vgg` (réseaux
profonds pré-entraînés), `nlpd` (pyramide laplacienne normalisée, modèle
de la rétine), `pamse`, `mad`, `vsi` (saillance visuelle), `reco`
(cohérence de phase entre deux harmoniques circulaires de Laguerre-Gauss —
Baroncini et al. 2009).

**Implémentation** : `fr/perceptual.py`, `fr/ssim.py`. `vif` défaut `sewar`
(pas de torch nécessaire) ; `gmsd` défaut `numpy` (vendored depuis sporco,
BSD-3-Clause, voir `backends/gmsd/numpy_impl.py`) ; `issm` défaut `numpy`
(`backends/issm/numpy_impl.py`, voir ci-dessous) ; `fsim`/`sam`/`sre`
défaut `image-similarity-measures` ; `reco` défaut `numpy`
(`backends/reco/numpy_impl.py`).

**`dists`/`vsi`/`lpips_vgg` ont pour défaut `piq`** (Apache-2.0,
`backends/piq/common.py`) et nécessitent l'extra `deep` ;
`fsim`/`gmsd`/`vif`/`ssim`/`msssim` ont `piq` comme algo optionnel
supplémentaire (défauts principaux inchangés, déjà légers). `piq` est
utilisé plutôt que IQA_pytorch (non maintenu depuis ~2021, cassé avec
torch récent) partout sauf pour `nlpd` (seul algo, `IQA_pytorch`, legacy)
et `mad` (aucune alternative permissive identifiée, `'algo': []`) — `piq`
n'implémente ni l'un ni l'autre. `pyiqa` a été évalué et écarté : licence
PolyForm-Noncommercial-1.0.0, non open-source (voir
`THIRD_PARTY_LICENSES.md` §2.3).

**Limites** : `mad` reste **inutilisable quelle que soit la version
installée** (`config.py`, `'algo': []`) — `lpips_vgg`/`vsi` sont
fonctionnels via `piq`.
`dists`/`lpips_vgg` (`piq`) téléchargent les poids VGG16 pré-entraînés
(~528 Mo) au premier appel — nécessite un accès réseau une fois, mis en
cache ensuite (voir `backends/piq/common.py`). `msssim` (`piq`) requiert
des images d'au moins 161×161 pixels (5 échelles). `pamse` : seul algo
`numpy` (vendored sporco, BSD-3-Clause).

**`gmsd` et `issm` : les algos disponibles ne sont PAS interchangeables**
(contrairement au reste du registre, où `algo=` sélectionne une
implémentation alternative d'une même définition — voir aussi la note
équivalente [§3.3](no-reference.md#33-sharpness--façade-à-5-sous-familles) pour `sharpness`/`fmeasure`) :

- `gmsd` — `numpy` vs `piq`/`IQA_pytorch` (`piq`/`IQA_pytorch` quasi
  identiques entre eux) : écart mesuré de 3,4 % (image 256×256) à 23,7 %
  (2048×2048) — pas du bruit flottant. `numpy` est aussi 6 à 18× plus
  lent à grande échelle. Les deux côtés prétendent pourtant porter la même
  implémentation MATLAB de référence (Xue et al. 2014), ce qui rend
  l'écart d'autant plus suspect.
- `issm` — `numpy` vs `image-similarity-measures` : écart mesuré de ~1 %
  (2048×2048) à ~11 % (256×256). Ici l'écart est **voulu** : `numpy`
  réimplémente ISSM d'après une lecture de l'article jugée plus fidèle
  (seule à reproduire ses figures, voir docstring du module) —
  `image-similarity-measures` est considéré moins fidèle, pas équivalent.
  `numpy` reste ~3× plus lent.

Un avertissement (`logger.warning`, une fois par process) est émis au
premier appel de `gmsd()`/`issm()` pour rappeler que les scores obtenus
avec des algos différents ne se comparent pas.

**Plan d'action pour trancher** (non engagé, voir aussi l'annexe en fin de
document) :
1. Reconstruire un cas de test à résultat connu (image de référence +
   dégradation + score publié dans l'article source ou dans une
   implémentation MATLAB/Python faisant autorité) pour chaque métrique.
2. Comparer chaque algo disponible à ce résultat de référence plutôt
   qu'entre eux — l'algo le plus proche devient le défaut, les autres
   restent disponibles mais documentés comme divergents (statu quo
   actuel) ou sont retirés si l'écart s'avère être une erreur
   d'implémentation identifiée.
3. Pour `gmsd` spécifiquement : vérifier l'hypothèse la plus probable
   (différence de sous-échantillonnage/mise à l'échelle avant calcul du
   gradient, point connu de divergence entre portages de l'algorithme de
   Xue et al.) en comparant les cartes intermédiaires (`return_map=True`)
   plutôt que seulement le score final.

### 2.6 Spectrale / télédétection

**Méthode** : erreur multi-bandes pensée pour l'imagerie satellite —
`ergas` (erreur globale adimensionnelle, pondérée par résolution),
`rase` (erreur spectrale relative moyenne), `sam` (angle spectral entre
vecteurs de réflectance), `sre` (signal-to-reconstruction error).

**Implémentation** : `fr/spectral.py`. Tous via `sewar` ou
`image-similarity-measures`, un seul algo chacun.

**Limites** : `rase` restreint à GRAY ; `sam` restreint aux images
multicanal (l'angle entre deux scalaires positifs est toujours nul) : il
renvoie un angle moyen en degrés, plus petit = meilleur, calculé en
float64 (l'arc cosinus est mal conditionné près de 1). Ces métriques
supposent des bandes alignées/coregistrées — aucune vérification de
recalage n'est faite en amont (à la charge de l'appelant).

**Enveloppes `image-similarity-measures`** (`fsim`, `issm`, `sam`, `sre`,
`psnr`, `rmse`, `ssim`, `uqi`) : les images sont passées au format
`(H, W, C)` par `core.checker.as_channels_last` (axe de canal ajouté aux
seules images 2D).

**Dynamique des images.** Plusieurs fonctions d'`image-similarity-measures`
supposent des images 12 bits (dynamique 4095), quelle que soit l'image
fournie ; sans correction, leur résultat dépend de l'échelle des
valeurs (ex. PSNR de 52 dB pour une image 0-255 et 100 dB pour la même
image en 0-1, au lieu de 28 dB). Les enveloppes fixent donc cette
dynamique :

| Métrique | Hypothèse amont | Correction | Résultat |
|---|---|---|---|
| `psnr`, `rmse` | `max_p = 4095` | `max_p` = dynamique de la référence (convention de l'algo `numpy`) | `psnr` identique à l'algo `numpy` ; `rmse` normalisée par la dynamique |
| `ssim` | `data_range = 4095` | `max_p` = dynamique conjointe des deux images (convention de l'algo `skimage`) | identique à l'algo `skimage` |
| `fsim` | `T2 = 160`, fixé pour des images 8 bits | images ramenées sur [0, 255] | indépendant de l'échelle |
| `issm` | image × 0.0625 avant Canny à seuils fixes 100/200, SSIM sur 4095 | images ramenées sur [0, 4095] | 1 pour deux images identiques au lieu de 0 partout |

`uqi`, `sam` et `sre` sont indépendants de l'échelle. Les tests vérifient
l'indépendance à l'échelle de chaque enveloppe concernée
(`tests/test_ism_wrappers.py`).

**`issm` (algo `numpy`, défaut)** — Aljanabi, Hussain, Shnain & Lu
(2019), *European Journal of Remote Sensing* 52(sup4), 2-15,
doi:10.1080/22797254.2019.1628617. Éq. (16) :
`I = (C·EHS·(a+b) + e) / (a·C·EHS + b·EHS + c·S + e)`, a, b, c = 0.3,
0.5, 0.7 ; C = |corrélation| des cartes de Canny (éq. 14) ; EHS =
entropie du histogramme conjoint 256 × 256 (éq. 13) ; S = SSIM (L = 255).
Choix là où l'article est muet ou incohérent :

- *EHS* : les équations (10)-(13) normalisent le histogramme
  (probabilités), mais cette lecture donne ~0.92 pour deux images
  identiques et ~0.3 pour deux images sans rapport, ce que contredisent
  les figures 3-10 de l'article (→ 1 et ≈ 0). Les figures ne sont
  reproduites qu'en **comptes** : défaut `ehs="counts"` ;
  `ehs="probabilities"` suit les équations à la lettre.
- *e* : non défini dans l'article ; nombre d'Euler.
- *Canny* : seuils automatiques à la manière de MATLAB `edge(I,'canny')`
  (outil des auteurs) : σ = √2, seuil haut = 70e centile de la norme du
  gradient, seuil bas = 0.4 × seuil haut.
- *Dynamique* : images ramenées sur [0, 255] via `data_range` (défaut :
  dynamique de la référence) — résultat indépendant de l'échelle des
  valeurs.

Comportement (image `camera`, bruit gaussien) : 0.99 à 60 dB, 0.75 à
30 dB, 0.31 à 20 dB, 0.11 à 10 dB ; < 0.05 entre images sans rapport.
L'algo `image-similarity-measures` (images ramenées sur [0, 4095]) suit
la même tendance mais décroît plus vite (0.69 à 30 dB, 0.19 à 20 dB sur
une image 256 × 256) : seuils de Canny fixes, histogramme à 10 classes,
corrélation des contours sans valeur absolue.
**Limite** : en comptes, EHS (négative, en N log N) écrase les autres
termes et I ≈ 0.8·C/(0.3·C + 0.5) : l'ISSM se comporte comme une
similarité de contours, le SSIM ne pesant presque rien ; pour deux
images identiques, I = 1 + 0.7/(0.8|EHS|) (léger dépassement de 1 sur
petite image). Niveaux de gris uniquement.
