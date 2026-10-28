# Référence technique — eoqual-metrics

Document de traçabilité de l'ensemble des métriques cataloguées dans
`eoqual-metrics`. Corps principal centré sur l'**état actuel** : **méthode**
(que calcule-t-on, indépendamment du code) → **implémentation** (où c'est
codé, avec quoi, quel est le défaut) → **limites connues**. La feuille de
route (non engagée) est reportée en annexe, pour que le corps reste une
référence sur ce qui existe aujourd'hui plutôt qu'un historique de projet.

Ce document ne remplace pas les docstrings (référence API précise,
autoapi/Sphinx), `BIBLIOGRAPHY.md` (référence bibliographique complète de
chaque méthode nommée — ce document-ci ne garde qu'un rappel court
auteur/année) ni `THIRD_PARTY_LICENSES.md` (audit des licences tierces,
document séparé) : il sert de carte d'ensemble pour comprendre *pourquoi*
chaque famille existe et *où* la retrouver — à maintenir à chaque
évolution notable du registre `config.py`.

## Sommaire

1. [Architecture générale](#1-architecture-générale)
2. [Full-Reference (FR) — comparer deux images](#2-full-reference-fr)
3. [No-Reference (NR) — évaluer une image seule](#3-no-reference-nr)
4. [Reduced-Reference (RR)](#4-reduced-reference-rr)
5. [Limites transverses](#5-limites-transverses)
6. [Annexe — Évolutions envisagées](#annexe--évolutions-envisagées)

---

## 1. Architecture générale

Un unique registre, `config.py::METRICS_CONFIGS`, décrit chaque métrique :
ses algorithmes disponibles (`algo`), l'algorithme par défaut (`default`),
le mode (`FR`/`NR`), le sens d'optimalité (`lower_better`), et la
restriction éventuelle de bande (`only: 'GRAY'|'RGB'`). Trois façons de
consommer ce registre :

- **Appel direct** : `eoqual.metrics.<nom_métrique>(GT, P, algo=...)` (FR)
  ou `(P, algo=...)` (NR).
- **Façade multi-algo** : certaines familles (`sharpness`) exposent une
  seule fonction publique qui dispatché en interne vers un backend selon
  `algo=`.
- **Orchestrateur `MetricsRunner`** (`runner.py`) : construit un plan
  d'exécution depuis le registre (ou un sous-ensemble choisi), calcule sur
  une ou plusieurs paires, accumule, exporte (`export/console.py`,
  `export/csv.py`).

Convention de code : chaque métrique/algo vit dans son propre module sous
`backends/<famille>/<impl>.py` (FR) ou
`nr/sharpness/backends/<famille>/<algo>.py` (NR sharpness) ; la fonction
publique de plus haut niveau (`fr/*.py`, `nr/*.py`) ne fait que
sélectionner le bon backend et normaliser les entrées/sorties
(`core/checker.py::initial_check`, `core/image_utils.py`).

---

## 2. Full-Reference (FR)

Comparent une image traitée `P` à une référence `GT` (même shape).

### 2.1 Erreur pixel-à-pixel

**Méthode** : écart brut entre pixels homologues — moyenne (`mae`, `mse`),
racine (`rmse`), médiane (`mdae`), version normalisée par la dynamique
(`nrmse`), ou calculée par fenêtre glissante (`rmse_sw`).

**Implémentation** : `fr/error.py`. `mae`/`mdae`/`mse`/`rmse` : `numpy`
seul algo. `nrmse` : `skimage`, seul algo. `rmse_sw` : `sewar`, seul algo.
`mse`/`rmse` acceptaient auparavant d'autres bibliothèques
(`skimage`/`metrikz`/`sewar`/`image-similarity-measures`) — retirées
après validation (résultat bit-exact, temps de calcul équivalent sur
image réelle et synthétique ; détail dans `formulas.md`), sauf
`image-similarity-measures` pour `rmse`, qui calculait en réalité une
*autre* métrique (RMSE normalisée par la dynamique, pas la RMSE brute —
écart ~99,6 %, cohérent avec une division par ~255). Appeler ces
fonctions avec un ancien nom d'algo reste accepté (compatibilité
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
`skimage`/`metrikz`/`sewar`/`image-similarity-measures` retirés après
validation (résultat bit-exact, temps de calcul équivalent ; détail dans
`formulas.md`) ; un ancien nom d'algo reste accepté (avertissement, calcul
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
7 algos alternatifs dont 4 vendored (`ssim_1`..`ssim_4`, voir §5.2 licences),
plus `piq` (Apache-2.0, GRAY+RGB) et `IQA_pytorch` (RGB,
legacy). `msssim` défaut `sewar`, alt. `piq` (≥161×161 px). `cw_ssim`
défaut `cw_ssim_1` (seul algo actif, GRAY uniquement, MIT — voir
`backends/cw_ssim/wavelet_impl.py`).

**Limites** : les algos `piq`/`IQA_pytorch` de `ssim`/`msssim` nécessitent
l'extra `deep` (torch). `cw_ssim` n'a pas d'implémentation RGB active
(celle IQA_pytorch a été retirée du registre, cassée ; `piq` n'implémente
pas CW-SSIM). Pas de version par tuiles pour la famille SSIM (calcul
toujours pleine image — voir §3.4 pour le tuilage NR).

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
`sklearn` : retirés après validation (résultat quasi bit-exact, écart
< 2e-13 ; temps de calcul équivalent ; détail dans `formulas.md`) — un
ancien nom d'algo reste accepté (avertissement, calcul fait avec
`numpy`). `scikit-learn` n'était utilisé nulle part ailleurs dans le
projet : retiré des dépendances. `nmi` : seul algo `skimage`. `scc`/`uqi` :
`sewar` (+ alternatives pour `uqi`, voir §5.2).

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
désormais fonctionnels via `piq` (revenus d'un état cassé de longue date).
`dists`/`lpips_vgg` (`piq`) téléchargent les poids VGG16 pré-entraînés
(~528 Mo) au premier appel — nécessite un accès réseau une fois, mis en
cache ensuite (voir `backends/piq/common.py`). `msssim` (`piq`) requiert
des images d'au moins 161×161 pixels (5 échelles). `pamse` : seul algo
`numpy` (vendored sporco, BSD-3-Clause).

**`gmsd` et `issm` : les algos disponibles ne sont PAS interchangeables**
(contrairement au reste du registre, où `algo=` sélectionne une
implémentation alternative d'une même définition — voir aussi la note
équivalente §3.3 pour `sharpness`/`fmeasure`) :

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
fournie ; sans correction, leur résultat dépendait de l'échelle des
valeurs (ex. PSNR de 52 dB pour une image 0-255 et 100 dB pour la même
image en 0-1, au lieu de 28 dB). Les enveloppes fixent désormais cette
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

---

## 3. No-Reference (NR)

Évaluent une image seule, sans vérité terrain.

### 3.1 Blind IQA (BIQA)

**Méthode** : scores appris statistiquement sur de larges bases d'images
notées par des humains — `brisque` (statistiques MSCN + régression SVM),
`niqe` (distance à un modèle de "naturalité" appris sur images nettes,
sans apprentissage supervisé du jugement humain — d'où "natural"), `piqe`
(segmentation par blocs + statistiques locales, sans apprentissage).

**Implémentation** : `nr/brisque.py`, `nr/niqe.py`, `nr/piqe.py`. Défaut
`numpy` pour les trois (`backends/{brisque,niqe,piqe}/numpy_impl.py`) ;
`niqe`/`brisque` partagent le prétraitement MSCN/AGGD via `core/nss.py`.
`brisque` conserve 2 algos alternatifs (`image-quality`, `brisque`), tous
deux nécessitant l'extra `svm`.

**Limites** : les modèles pré-entraînés (`niqe/models/modelparameters.mat`,
`brisque/models/allmodel`) sont conservés depuis la release officielle
LIVE Lab (licence de recherche permissive, voir `THIRD_PARTY_LICENSES.md`
et `models/NOTICE.md`) — ce sont des paramètres statistiques appris, pas
du code réécrivable. `niqe` nécessite une image d'au moins 193×193 pixels
(granularité de bloc du modèle). La prédiction SVR de `brisque` recharge
le fichier modèle à chaque appel (voir §6).

### 3.2 Focus / entropie

**Méthode** : `entropy` (richesse d'information de l'histogramme —
image floue = distribution resserrée). `fmeasure` : 27 opérateurs de
mise au point issus de la littérature *shape-from-focus* (Pertuz et al.
2013) — gradients (BREN, TENG, GRAT...), Laplaciens (LAPE, LAPM, LAPV...),
ondelettes (WAVR, WAVS, WAVV), statistiques (GLVA, GLVN...), etc. Comme
pour `sharpness` (§3.3), `algo=` sélectionne ici une méthode de mesure
différente pour chaque valeur, pas une implémentation alternative d'une
même définition — les 27 opérateurs ne sont pas interchangeables entre
eux ; voir `BIBLIOGRAPHY.md`.

**Implémentation** : `nr/focus.py` (`entropy`), `nr/focus_fmeasure.py`
(`fmeasure`, `MEASURES_FOCUS` liste les 27 noms). Défaut `ACMO`. Chaque
opérateur est implémenté depuis sa définition mathématique publiée (voir
la docstring de chaque fonction privée du module pour sa référence
précise).

**Limites** : le comportement de `DCTR` sur un bloc 8×8 incomplet en bord
d'image (dimensions non multiples de 8) lève une exception — comportement
préexistant, reproduit à l'identique, non corrigé.

### 3.3 Sharpness — façade à 5 sous-familles

Point d'entrée unique `nr.sharpness.sharpness(P, algo=...)`, 17 algos.

**Attention, `algo=` a ici un sens différent du reste du registre** : pour
la plupart des métriques (`psnr`, `ssim`...), `algo=` sélectionne une
implémentation alternative d'une même définition mathématique — les
valeurs sont interchangeables. Pour `sharpness` (et `fmeasure`, §3.2),
chaque `algo=` est une **méthode de mesure différente** : deux algos
peuvent classer deux images dans un ordre différent. Voir
`BIBLIOGRAPHY.md` pour la référence de chaque méthode, et la docstring du
module pour les cas de faux-amis avec `fmeasure` (même filiation
historique, formule différente : `tenengrad_otsu`/`TENG`,
`laplacian`/`LAPV`, `brenner_vertical`/`BREN`).

Audit de collision de noms effectué sur les 17×27 combinaisons
`sharpness`×`fmeasure` (comparaison des formules, pas seulement des noms) :
seuls ces trois cas partagent un nom suffisamment proche pour induire en
erreur — les autres proximités conceptuelles (`sobel` vs `GRAT`/`GRAS`/
`GRAE`, `wavelet` vs `WAVR`/`WAVS`/`WAVV`, `s3`/`mtf` vs `SFIL`/`SFRQ`)
utilisent des formules ET des noms suffisamment distincts pour ne pas
prêter à confusion.

#### 3.3.a Gradient

**Méthode** : énergie/amplitude des contours détectés par filtre de
Sobel (`tenengrad_otsu`, Krotkov 1987, seuillage Otsu ; `sobel`) ou
variance du Laplacien (`laplacian`, Pech-Pacheco et al. 2000 — répond aux
variations d'intensité dans toutes les directions simultanément).

**Implémentation** : `backends/gradient/{tenengrad,laplacian,sobel}.py`.
Originaux (pas de code vendored), défaut `tenengrad_otsu`.

**Limites** : sensibles au bruit (un bruit fort produit lui-même un fort
gradient local, indiscernable d'un vrai contour net) et au contenu de la
scène (non comparables entre scènes différentes — voir §3.3.e pour
l'alternative calibrée).

#### 3.3.b Phase

**Méthode** : cohérence de la phase locale plutôt que l'amplitude du
gradient — plus robuste au contraste/illumination. `lpc_si` (Local Phase
Coherence, Kovesi/Wang), `sharpness_index` (variation totale, Moisan/
Blanchet), `psi` (Perceptual Sharpness Index, Feichtenhofer 2013 —
distingue flou horizontal/vertical).

**Implémentation** : `backends/phase/{lpc_si,sharpness_index,psi}.py`.
Réimplémentations originales d'Olivier Amram à partir des publications
citées (pas de portage de code tiers) — GRAY uniquement.

**`sharpness_index` en détail** : compare la variation totale TV(u) à
celle d'un champ gaussien de même spectre d'amplitude et de phase
aléatoire, dont la moyenne μ et l'écart-type σ sont calculés en forme
close ; `SI = -log10 Φ((μ - TV)/σ)`. Prétraitement `pmode=3` par défaut :
composante périodique (Moisan 2011, supprime les faux contours de bord)
puis déquantification (translation d'un demi-pixel). Calcul en float64 ;
`sharpness_stats` expose μ, σ et z. Les moments sont validés contre une
simulation Monte-Carlo (`tests/test_sharpness_index.py`).

**Limites** : `lpc_si` et `psi` n'ont pas de suite de tests de
non-régression contre une implémentation de référence publiée. SI est un
score ordinal : à comparer entre versions d'une même scène, à taille
égale.

#### 3.3.c Spectral

**Méthode** : contenu fréquentiel de l'image. `s3` (Spectral and Spatial
Sharpness — combine variation totale locale et pente du spectre de
Fourier, percentile 90 en agrégation). `mtf` : MTF "légère" par arêtes
ponctuelles (Canny + profil de gradient local + FFT du LSF moyen) — donne
un vrai contraste en cycles/pixel, mais sans filtrage géométrique
d'arête (angle/rectitude) contrairement à `aem`/`sasbem` (§3.3.e).

**Implémentation** : `backends/spectral/{s3,mtf}.py`. `mtf` implémente la
méthode décrite ci-dessus en une seule version — pas de doublon avec
`aem`/`sasbem`.

**Limites** : `mtf` renvoie `insufficient=True`/score 0.0 sur une scène
sans arête exploitable (comportement voulu, pas un bug). `s3` : seuil de
blocs fixe (32×32), non paramétrable depuis la façade.

#### 3.3.d Perceptuelle (librairie externe)

**Méthode** :
- `cpbd` (Cumulative Probability of Blur Detection, Narvekar & Karam
  2011 — modèle psychophysique de détection du flou aux contours).
- `blur_effect` (Crété, Dolmière, Ladret & Nicolas 2007 — reflou par
  filtre moyenneur 1-D, comparaison de la variation entre pixels voisins
  avant/après reflou, par axe ; validé par tests subjectifs ITU-R). Délègue
  à `skimage.measure.blur_effect` (même publication citée dans sa
  docstring) plutôt qu'une réimplémentation — scikit-image est déjà une
  dépendance de base. Score inversé (`1 - score brut`) pour rester
  cohérent avec la convention "plus élevé = plus net" du reste de la
  façade `sharpness` (le score brut du papier est l'inverse : 0=net,
  1=flou) — voir docstring de `blur_effect()` pour le détail et la
  validation croisée contre `deepinv.loss.metric.BlurStrength`
  (BSD-3-Clause, même référence bibliographique).

**Implémentation** : `backends/perceptual/cpbd.py`,
`backends/perceptual/blur_effect.py`.

**Limites** : `cpbd` — dépendance non maintenue, shim de compatibilité
(voir §3.3.f). `blur_effect` — asymptote au-delà d'un flou moyen
équivalent au filtre `filter_size` (11 par défaut) : au-delà, la métrique
reste correcte dans son classement mais moins discriminante (voir
docstring de `skimage.measure.blur_effect`).

#### 3.3.e Satellite (calibré, remote sensing)

**Méthode** : seule sous-famille physiquement calibrée et comparable
entre capteurs/dates/missions (hors métriques génériques ci-dessus,
sensibles au contenu de la scène) :

- `brenner_vertical`, `fft`, `wavelet` : indicateurs rapides non calibrés
  (variations abruptes espacées, énergie haute fréquence, énergie des
  sous-bandes en ondelettes) — mêmes limites que §3.3.a/c, mais issus
  d'un jeu d'implémentations pensé pour l'imagerie EO.
- `antonel` : décroissance de gradient normalisée avant/après flou de
  référence, par axe (Sx/Sy) — robuste au bruit/exposition, diagnostic
  de flou anisotrope (bougé). Implémentation clean-room fidèle à Antonel
  (2024) — à l'exception de l'indicateur de représentativité du papier
  (§3.6), non implémenté (voir docstring du module).
- `blur_kernel` : reconstruction d'une PSF 1-D moyenne par profils
  d'arêtes, mesure sa largeur à mi-hauteur (FWHM). Reprend le principe de
  scoring (norme du noyau) de Anger et al. (2019) — mais avec une
  estimation de noyau bien plus simple que leur déconvolution aveugle
  itérative (voir docstring).
- **`aem`** (Automatic Edge Method) : méthode du bord incliné
  (*slanted-edge*, ISO 12233) appliquée à des arêtes naturelles
  auto-détectées (PCA : longueur, angle 2°-15°, rectitude), sans mire de
  calibration. Score = médiane de la MTF@Nyquist par arête retenue.
- **`sasbem`** : AEM + filtrage statistique robuste (MAD, facteur 1.4826)
  des arêtes aberrantes + indice de fiabilité (`n_kept / n_measured`).

**Implémentation** : `nr/sharpness/backends/satellite/` — `brenner.py`,
`fft.py`, `antonel.py`, `blur_kernel.py`, `wavelet.py`, `aem.py`,
`sasbem.py`, `_slanted_edge.py` (cœur PCA/ESF/LSF/FFT partagé par
`aem`/`sasbem`). Toutes GRAY uniquement, exposent `return_details=True`
pour le détail (mtf50, fwhm, n_edges, reliability, insufficient).

Dans `_slanted_edge.py`, la détection et la sélection des arêtes sont
propres à metrics ; le binning de l'ESF, la MTF (normalisée en fréquence
nulle), la MTF@Nyquist, la MTF50 et la FWHM viennent d'`eoqual.core.edge`
(partagé avec eoqual-measure). La FWHM est calculée sur la LSF lissée par
une gaussienne d'un demi-pixel, puis par la première traversée du
demi-maximum de part et d'autre du pic : l'ancienne définition (premier
et dernier point au-dessus du seuil sur tout le profil) dépassait 15 px
sur un simple rebond de bruit.

**Limites** :
- Ces métriques ne sont analysées qu'en image entière par défaut — pour
  une scène large où un score unique masquerait une hétérogénéité locale,
  voir le calcul par tuiles opt-in (§3.4).
- Arêtes rejetées avant la médiane (`detect_and_measure_edges`) : LSF
  qui s'annule presque (`|Σ LSF| / Σ |LSF| < 0.1` : ligne fine plutôt que
  bord) ou MTF > 1.5 avant Nyquist (lobes négatifs : arête voisine, coin,
  texture dans la bande de projection). Sur l'image `r0`, 4 arêtes sur 8
  rejetées (dont une à MTF@Nyquist = 73) : `aem` passe de 0.167 à 0.050.
  Sur scènes synthétiques, la médiane se rapproche de la MTF vraie dans
  9 cas sur 11. Seuils réglables (`max_mtf_overshoot`,
  `min_edge_sign_ratio`).
- Plancher de bruit : même après ce rejet, la MTF@Nyquist par arête
  reste au-dessus de la valeur vraie sur scène floue et bruitée (ex. 0.15
  pour 0.06 attendu) — le bruit de l'ESF ne s'annule pas à haute
  fréquence. À interpréter comme un indicateur relatif, pas comme une
  mesure absolue (voir eoqual-measure pour une mesure sur mire).
- `aem`/`sasbem` renvoient `0.0`/`insufficient=True` sur les scènes
  homogènes (eau, forêt dense, désert) — résultat attendu, pas une erreur
  (pas d'arête rectiligne exploitable), mais peut surprendre à l'usage si
  non documenté en aval.
- Pas de contrôle du *Ground Sampling Distance* (GSD) : comparer un score
  MTF entre un produit natif et un produit ré-échantillonné/pansharpené
  sans en tenir compte n'a pas de sens physique (limite inhérente à la
  méthode, pas à l'implémentation).
- Pas de suite de tests dédiée (au-delà du smoke-test manuel effectué à
  l'introduction de ces métriques — voir §6, tests automatisés à ajouter).

#### 3.3.f Dépendance cpbd

`cpbd` (Narvekar & Karam) n'est plus maintenu depuis 2018 ; une ancienne
version importe `from scipy.ndimage import imread`, retiré de SciPy
≥ 1.3. `backends/perceptual/cpbd.py::_ensure_cpbd_importable()` fournit un
stub `scipy.ndimage.imread` avant l'import de `cpbd`, quelle que soit la
version installée — aucune étape d'installation manuelle requise.

### 3.4 Calcul par tuiles (No-Reference, opt-in)

**Méthode** : sur une scène large, un score unique moyenne toute
hétérogénéité locale (flou de bougé localisé, zone nuageuse, plage d'eau
sans arête exploitable pour `aem`/`sasbem`). `tiling.compute_score_on_tiles`
découpe l'image en tuiles chevauchantes et applique n'importe quelle
métrique NR à chacune, produisant une carte de scores avec un indice de
représentativité par tuile — repris de l'ancien moteur de tuilage de
`sharpness_metrics`.

**Implémentation** : `eoqual/metrics/tiling.py`, exposé aussi comme
`eoqual.metrics.compute_score_on_tiles`. Générique : accepte toute fonction
`f(image, **kwargs) -> float` ou `f(image, **kwargs) -> (float, dict)` —
un backend direct (`aem`, `sasbem`, `mtf`, `antonel`, `blur_kernel`, avec
`method_kwargs={"return_details": True}` pour l'indice de fiabilité natif)
ou n'importe quelle métrique figée par `functools.partial` (ex.
`partial(m.sharpness, algo="tenengrad_otsu")`). Représentativité par défaut :
variance de la tuile ; recouvrement automatique par `n_edges`/`reliability`
pour `aem`/`sasbem`/`mtf`, par `1/fwhm` pour `blur_kernel`. Log résumé
unique par (image, métrique) plutôt qu'un message par tuile.

**Volontairement non implémenté à ce stade** (documenté en tête de
`tiling.py`) :
- **Parallélisation** (`ThreadPoolExecutor` sur les tuiles) : `aem`/
  `sasbem` sont encore à un stade de développement — paralléliser avant
  leur stabilisation ajouterait de la complexité (erreurs concurrentes,
  logs non déterministes) pour un gain prématuré. Le point d'insertion
  serait la boucle `for` de `compute_score_on_tiles`.
- **Intégration à `MetricsRunner`** : le tuilage reste une fonction
  autonome, jamais appelée implicitement — changer le contrat de résultat
  du runner (un score, pas une carte) est un chantier séparé.
- **Full-Reference** : ce module ne couvre que les métriques NR (une
  image). Pas de tuilage FR dédié (les cartes pixel-à-pixel `return_map=True`
  existantes sur certaines métriques FR restent le mécanisme équivalent).

**Limites** : découpage rectangulaire simple (pas de padding — une image
plus petite que `tile_size` dans une dimension donne une seule tuile sur
cette dimension). Pas de suite de tests dédiée au-delà de la validation
manuelle effectuée à l'introduction (cas limites : image plus petite que
la tuile, dimensions non multiples de `tile_size`, `overlap=0`).

---

## 4. Reduced-Reference (RR)

**Aucune métrique implémentée à ce jour** (README, section « Reduced-Reference
IQA » : *None*). Catégorie documentée dans le registre conceptuel mais pas
dans `METRICS_CONFIGS` — à considérer si un besoin se présente (ex.
comparaison à un jeu de statistiques de référence plutôt qu'à l'image
complète, utile quand le GT complet n'est pas disponible/transmissible).

---

## 5. Limites transverses

### 5.1 Gestion des erreurs du runner

`MetricsRunner.compute()` (`runner.py`) laisse volontairement remonter
toute exception (`try/except` commenté) tant que la librairie n'est pas
jugée stable — choix assumé du mainteneur : une trace complète
(fichier/ligne/cause) est plus utile pour diagnostiquer qu'un `value=None`
silencieux. Conséquence pratique : un batch multi-images s'arrête net à
la première métrique qui échoue (dépendance manquante, image incompatible...).
À réévaluer une fois la robustesse sur de gros lots jugée prioritaire par
rapport au diagnostic fin (voir l'annexe en fin de document).

### 5.2 Licences tierces

Voir **`THIRD_PARTY_LICENSES.md`** pour l'audit complet. Aucun algorithme
obtenu par défaut n'est concerné : les quelques algorithmes 🟡 sous licence
GPL (`metrikz` pour `wsnr`/alt., `ssim_1..4`) sont tous optionnels,
jamais atteints sans les demander explicitement via `algo=`.

### 5.3 Dépendances lourdes / fragiles

`torch`/`torchvision`/`piq`/`IQA_pytorch` : extra optionnel `deep`, version
torch bornée (`<2.3.1`) sur la base des limitations documentées d'
IQA_pytorch (torch ≥ 2.3.1 casse son algo `cw_ssim`, non par défaut).

`piq` (Apache-2.0) est le backend par défaut pour
`dists`/`fsim`/`gmsd`/`vif`/`ssim`/`msssim`/`vsi`/`lpips_vgg` — voir
§2.3/§2.5 et `THIRD_PARTY_LICENSES.md`. IQA_pytorch (non maintenu depuis
~2021, MIT) reste disponible en algo legacy partout, et c'est le seul
backend pour `nlpd`. `pyiqa` a été évalué comme alternative plus complète
(couvre aussi `nlpd`/`mad`) et écarté : licence
PolyForm-Noncommercial-1.0.0, non open-source (voir
`THIRD_PARTY_LICENSES.md` §2.3).

### 5.4 Couverture de tests

Trois catégories distinctes, à ne pas confondre :

- **`examples/`** — usage documenté, à jour, destiné à être lu/exécuté par
  un utilisateur : `all_metrics_overview.py` (registre complet),
  `one_metric_cli.py` (CLI une métrique/une image), `tiled_vs_full_image.py`
  (comparaison calcul pleine image vs. par tuiles, voir §3.4).
- **`tests/exploratory/`** — scripts d'investigation ponctuelle, pas
  maintenus, pas des exemples à suivre (API tierce appelée directement,
  hors registre eoqual-metrics) : `recover_iqa_pytorch.py`,
  `sewar_overview.py`, `pyiqa_rejected_exploration.py` (documente pourquoi
  `pyiqa` a été écarté, voir `THIRD_PARTY_LICENSES.md` §2.3).
- **`tests/`** — ne garde que `images/` (fixtures) et le vérificateur de
  cohérence `__all__`/`METRICS_CONFIGS` embarqué dans
  `examples/all_metrics_overview.py`. Réservé aux futurs tests `pytest`
  réels — voir `CONTRIBUTING.md` pour contribuer une suite.

**Une vraie suite `pytest` n'existe pas encore**, volontairement : le
registre continue d'évoluer (`aem`/`sasbem` encore en développement), et
une suite assertive écrite maintenant demanderait
d'être retouchée à chaque évolution de ce type — un coût d'entretien
prématuré par rapport à la valeur immédiate. `quality/run_tests.sh` reste
pointé sur `tests/` (commentaire à jour dans le script) : il ne collecte
rien pour l'instant, sans être cassé pour autant — prêt à l'emploi le jour
où une suite y sera ajoutée. Le workflow CI
(`.github/workflows/docs.yml`) construit et publie la documentation
(Sphinx + autoapi) sur GitHub Pages à chaque push sur `main`, mais ne
lance aucun test.

---

## Annexe — Évolutions envisagées

Feuille de route, pas un état des lieux : contrairement au corps du
document ci-dessus, rien ici ne décrit le comportement actuel de la
librairie. Par ordre de valeur/risque décroissant, à discuter et prioriser
avec le mainteneur — rien ci-dessous n'est engagé. Voir `CONTRIBUTING.md`
pour proposer ou implémenter l'un de ces points, ou tout autre bug/métrique :

1. **Remédiation licences des algorithmes optionnels restants** — les 5
   algorithmes 🟡 non par défaut (`metrikz`, `ssim_1..4`, voir
   `THIRD_PARTY_LICENSES.md` §2.1) pourraient être réécrits en clean-room
   ou retirés — impact nul sur l'usage standard, non engagé.
2. **Alternative permissive pour `nlpd`/`mad`** — `piq` ne les implémente
   pas ; `nlpd` reste fonctionnel via IQA_pytorch (legacy), `mad` reste
   inutilisable quelle que soit la dépendance (`'algo': []`).
3. **Parallélisation du calcul par tuiles** (`ThreadPoolExecutor`,
   `tiling.py`) — différée tant qu'`aem`/`sasbem` ne sont pas stabilisées
   (voir la docstring du module pour le point d'insertion exact).
4. **Intégration du tuilage à `MetricsRunner`** — actuellement une
   fonction autonome jamais appelée implicitement ; changer le contrat de
   résultat du runner (un score, pas une carte) est un chantier séparé.
5. **Suite de tests assertive** par métrique (valeurs de référence
   connues, cas limites — image constante, bruit pur, arêtes absentes) ;
   CI exécutant effectivement `pytest` (voir §5.4).
6. **Réactivation du try/except du runner** (§5.1) une fois la robustesse
   sur de gros lots jugée prioritaire — actuellement report volontaire.
7. **Reduced-Reference** : famille à définir si un besoin se présente
   (§4).
8. **`brisque/numpy_impl.py`** recharge le modèle SVR à chaque appel — un
   cache module-level éviterait de reparser `allmodel` (770 vecteurs de
   support) à chaque image sur de gros lots.

---

*Document vivant — à mettre à jour à chaque évolution notable de
`config.py` ou du registre de backends.*
