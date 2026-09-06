# Référence technique — eoqual-metrics

Document de traçabilité de l'ensemble des métriques cataloguées dans
`eoqual-metrics`. Structure en quatre temps, dans cet ordre
volontairement : **méthode** (que calcule-t-on, indépendamment du code) →
**implémentation** (où c'est codé, avec quoi, quel est le défaut) →
**limites connues** → **évolutions envisagées**.

Ce document ne remplace pas les docstrings (référence API précise,
autoapi/Sphinx) ni `THIRD_PARTY_LICENSES.md` (audit des licences tierces,
document séparé) : il sert de carte d'ensemble pour comprendre *pourquoi*
chaque famille existe et *où* la retrouver — à maintenir à chaque
évolution notable du registre `config.py`.

## Sommaire

1. [Architecture générale](#1-architecture-générale)
2. [Full-Reference (FR) — comparer deux images](#2-full-reference-fr)
3. [No-Reference (NR) — évaluer une image seule](#3-no-reference-nr)
4. [Reduced-Reference (RR)](#4-reduced-reference-rr)
5. [Limites transverses](#5-limites-transverses)
6. [Évolutions envisagées](#6-évolutions-envisagées)

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

**Implémentation** : `fr/error.py`. Défauts tous `numpy` (calcul direct),
sauf `nrmse` (`skimage`, seul algo) et `rmse_sw` (`sewar`, seul algo).
Alternatives `skimage`/`metrikz`/`sewar`/`image-similarity-measures` pour
`mse`/`rmse`.

**Limites** : mesures non perceptuelles (une MSE faible ne garantit pas une
image visuellement fidèle) — voir §2.3/2.5 pour des alternatives
perceptuelles. `rmse_sw` : l'algo `map=True` de sewar est désactivé dans
`config.py` (erreur de type constatée, jamais corrigée).

### 2.2 Signal-based (SNR/PSNR)

**Méthode** : rapport entre puissance du signal et puissance de l'erreur,
en dB. `psnrb` ajoute une pénalité de blocking (compression JPEG-like).
`psnrc` = variante couleur.

**Implémentation** : `fr/signal.py`. Défaut `numpy` pour `snr`/`psnr`/`psnrc` ;
`sewar` seul algo pour `psnrb` ; `wsnr` (pondération perceptuelle par CSF —
Contrast Sensitivity Function de Mannos & Sakrison) défaut `numpy`
(`backends/wsnr/numpy_impl.py`), `metrikz` conservé en
algo legacy (voir `THIRD_PARTY_LICENSES.md`).

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

**Implémentation** : `fr/correlation.py`. Défauts `numpy`/`skimage`/`sewar`/`metrikz`
selon la métrique. `nmi` : seul algo `skimage`. `scc` : seul algo `sewar`.
`jsd` : seul algo `numpy`.

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
`dists`/`lpips_vgg` (réseaux profonds pré-entraînés), `nlpd` (pyramide
laplacienne normalisée, modèle de la rétine), `pamse`, `mad`, `vsi`
(saillance visuelle), `reco` (cohérence de phase entre deux harmoniques
circulaires de Laguerre-Gauss — Baroncini et al. 2009).

**Implémentation** : `fr/perceptual.py`, `fr/ssim.py`. `vif` défaut `sewar`
(pas de torch nécessaire) ; `gmsd` défaut `numpy` (vendored depuis sporco,
BSD-3-Clause, voir `backends/gmsd/numpy_impl.py`) ; `fsim`/`issm`/`sam`/`sre`
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

### 2.6 Spectrale / télédétection

**Méthode** : erreur multi-bandes pensée pour l'imagerie satellite —
`ergas` (erreur globale adimensionnelle, pondérée par résolution),
`rase` (erreur spectrale relative moyenne), `sam` (angle spectral entre
vecteurs de réflectance), `sre` (signal-to-reconstruction error).

**Implémentation** : `fr/spectral.py`. Tous via `sewar` ou
`image-similarity-measures`, un seul algo chacun.

**Limites** : `rase` restreint à GRAY. Ces métriques supposent des bandes
alignées/coregistrées — aucune vérification de recalage n'est faite en
amont (à la charge de l'appelant).

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
ondelettes (WAVR, WAVS, WAVV), statistiques (GLVA, GLVN...), etc.

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

#### 3.3.a Gradient

**Méthode** : énergie/amplitude des contours détectés par filtre de
Sobel (`tenengrad`, Krotkov 1987 ; `sobel`) ou variance du Laplacien
(`laplacian`, Pech-Pacheco et al. 2000 — répond aux variations
d'intensité dans toutes les directions simultanément).

**Implémentation** : `backends/gradient/{tenengrad,laplacian,sobel}.py`.
Originaux (pas de code vendored), défaut `tenengrad`.

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

**Limites** : pas de suite de tests de non-régression contre une
implémentation de référence publiée.

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

**Méthode** : `cpbd` (Cumulative Probability of Blur Detection, Narvekar
& Karam 2011 — modèle psychophysique de détection du flou aux contours).
Seul algo de cette sous-famille.

**Implémentation** : `backends/perceptual/cpbd.py`.

**Limites** : dépendance non maintenue, shim de compatibilité (voir §3.3.f).

#### 3.3.e Satellite (calibré, remote sensing)

**Méthode** : seule sous-famille physiquement calibrée et comparable
entre capteurs/dates/missions (hors métriques génériques ci-dessus,
sensibles au contenu de la scène) :

- `brenner`, `fft`, `wavelet` : indicateurs rapides non calibrés
  (variations abruptes espacées, énergie haute fréquence, énergie des
  sous-bandes en ondelettes) — mêmes limites que §3.3.a/c, mais issus
  d'un jeu d'implémentations pensé pour l'imagerie EO.
- `antonel` : décroissance de gradient normalisée avant/après flou de
  référence, par axe (Sx/Sy) — robuste au bruit/exposition, diagnostic
  de flou anisotrope (bougé). Implémentation clean-room fidèle à Antonel,
  "A Novel No-Reference Image Quality Metric for Assessing Sharpness in
  Satellite Imagery", arXiv:2410.10488, 2024 — à l'exception de
  l'indicateur de représentativité du papier (§3.6), non implémenté
  (voir docstring du module).
- `blur_kernel` : reconstruction d'une PSF 1-D moyenne par profils
  d'arêtes, mesure sa largeur à mi-hauteur (FWHM). Reprend le principe de
  scoring (norme du noyau) de Anger, de Franchis, Facciolo, "Assessing
  the Sharpness of Satellite Images: Study of the PlanetScope
  Constellation", IGARSS 2019 — mais avec une estimation de noyau bien
  plus simple que leur déconvolution aveugle itérative (voir docstring).
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

**Limites** :
- Ces métriques ne sont analysées qu'en image entière par défaut — pour
  une scène large où un score unique masquerait une hétérogénéité locale,
  voir le calcul par tuiles opt-in (§3.4).
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
`partial(m.sharpness, algo="tenengrad")`). Représentativité par défaut :
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
rapport au diagnostic fin (voir §6).

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
registre continue d'évoluer (§6, notamment `aem`/`sasbem` encore en
développement), et une suite assertive écrite maintenant demanderait
d'être retouchée à chaque évolution de ce type — un coût d'entretien
prématuré par rapport à la valeur immédiate. `quality/run_tests.sh` reste
pointé sur `tests/` (commentaire à jour dans le script) : il ne collecte
rien pour l'instant, sans être cassé pour autant — prêt à l'emploi le jour
où une suite y sera ajoutée. Le workflow CI
(`.github/workflows/docs.yml`) construit et publie la documentation
(Sphinx + autoapi) sur GitHub Pages à chaque push sur `main`, mais ne
lance aucun test.

---

## 6. Évolutions envisagées

Par ordre de valeur/risque décroissant, à discuter et prioriser avec le
mainteneur — rien ci-dessous n'est engagé. Voir `CONTRIBUTING.md` pour
proposer ou implémenter l'un de ces points, ou tout autre bug/métrique :

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
