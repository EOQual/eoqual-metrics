# No-Reference : évaluer une image seule

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
le fichier modèle à chaque appel (voir [§6](limites.md#6-pistes)).

### 3.2 Focus / entropie

**Méthode** : `entropy` (richesse d'information de l'histogramme —
image floue = distribution resserrée). `fmeasure` : 27 opérateurs de
mise au point issus de la littérature *shape-from-focus* (Pertuz et al.
2013) — gradients (BREN, TENG, GRAT...), Laplaciens (LAPE, LAPM, LAPV...),
ondelettes (WAVR, WAVS, WAVV), statistiques (GLVA, GLVN...), etc. Comme
pour `sharpness` (§3.3), `algo=` sélectionne ici une méthode de mesure
différente pour chaque valeur, pas une implémentation alternative d'une
même définition — les 27 opérateurs ne sont pas interchangeables entre
eux ; voir [bibliographie](bibliographie.md).

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
[bibliographie](bibliographie.md) pour la référence de chaque méthode, et la docstring du
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
blocs fixe (32×32), non paramétrable depuis la façade. `s3` présentait un
bug de normalisation (réétalement min-max par image, rendant le score non
monotone avec le flou sur une bonne partie de la plage testée) — corrigé,
voir la docstring du module pour le détail et un résidu mineur de
non-monotonicité entre une image nette et un flou d'environ 1 pixel.

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
demi-maximum de part et d'autre du pic : prendre le premier et le
dernier point au-dessus du seuil sur tout le profil donne plus de 15 px
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
  l'introduction de ces métriques — voir [§6](limites.md#6-pistes), tests automatisés à ajouter).

#### 3.3.f Dépendance cpbd

`cpbd` (Narvekar & Karam) n'est plus maintenu depuis 2018 ; sa dernière
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
représentativité par tuile.

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
