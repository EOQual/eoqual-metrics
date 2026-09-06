# Licences tierces — inventaire et statut

> **Avertissement.** Cet inventaire est tenu au meilleur effort. Si une
> erreur de licence y est identifiée (source mal vérifiée, statut incorrect,
> nouvelle source vendored non recensée), elle sera corrigée dès que
> signalée — et si nécessaire, l'implémentation concernée sera retirée
> plutôt que sa licence "arrangée" a posteriori. Pour signaler un problème,
> voir `CONTRIBUTING.md`.

Ce document recense, pour chaque fichier de `src/eoqual/metrics/` qui reprend ou
adapte du code publié ailleurs, sa source, la licence **réellement vérifiée**
de cette source, et le statut de compatibilité avec la licence MIT annoncée
pour `eoqual-metrics` (`license.txt`).

**Pourquoi ce document existe** : `license.txt` déclare tout le dépôt MIT ;
quelques fichiers vendored embarquent une licence différente — parfois
auto-déclarée (`__license__ = "GPL"`), parfois absente (dépôt d'origine sans
fichier LICENSE, donc "tous droits réservés" par défaut). Ce fichier rend
visible tout écart, fichier par fichier.

**État actuel** : tous les algorithmes obtenus **par défaut** (sans préciser
`algo=`) sont sous licence permissive (MIT/BSD-3/Apache-2.0) ou licence de
recherche permissive avec citation (LIVE Lab, UT Austin — voir
`backends/niqe/models/NOTICE.md` et `backends/brisque/models/NOTICE.md`).
Seuls des algorithmes **optionnels**, jamais obtenus sans les demander
explicitement via `algo=`, restent sous une licence différente — voir §2.

## 1. Légende

- 🔴 **Bloquant** — licence copyleft (GPL/LGPL) ou absente, et c'est
  l'implémentation **par défaut ou unique** de la métrique (un utilisateur
  qui ne touche à rien l'utilise). *Aucune entrée dans cet état actuellement.*
- 🟡 **À surveiller** — même problème de licence, mais l'algorithme est
  **optionnel** (`algo=` non par défaut) : un utilisateur ne l'obtient que
  s'il le demande explicitement.
- 🟢 **Compatible** — licence permissive (MIT/BSD/Apache) ou recherche
  permissive avec citation, compatible avec MIT moyennant la conservation
  de la notice d'origine.

## 2. Inventaire

### 2.1 Algorithmes optionnels sous licence non permissive (🟡)

| Fichier | Métrique / algo | Source citée | Licence vérifiée |
|---|---|---|---|
| `backends/metrikz/metrikz.py` | alt. de `wsnr`, `mse`, `rmse`, `snr`, `psnr`, `psnrc`, `ssim`, `uqi`, `vif` | Pedro Garcia Freitas, pymetrikz | **GPLv2 / GPL** — auto-déclaré en tête de fichier |
| `backends/ssim/video_impl.py` | `ssim` algo `ssim_3` | Alex Izvorski, aizvorski/video-quality | **GPL-3.0** |
| `backends/ssim/skimage_impl.py` | `ssim` algo `ssim_1`, `msssim` algo `msssim_1` | mubeta06/python + RenYang-home/HLVC | **GPL-3.0/LGPL-3.0** (mubeta06) ; HLVC = BSD-3-Clause |
| `backends/ssim/improved_impl.py` | `ssim` algo `ssim_2` | Helder C. R. de Oliveira | **GPL** — auto-déclaré |
| `backends/ssim/clermont_impl.py` | `ssim` algo `ssim_4` | Antoine Vacavant, ISIT/Clermont | **GPL** — auto-déclaré |

Ces 5 fichiers restent optionnels : ils ne sont jamais utilisés par défaut, et
un utilisateur ne les obtient qu'en demandant explicitement leur `algo=`.
Impact nul sur l'usage standard de la librairie — voir §3 pour la
remédiation envisageable.

### 2.2 Sources permissives confirmées (🟢)

| Fichier | Métrique / algo | Source | Licence |
|---|---|---|---|
| `backends/vif/mscale_impl.py` | `vif` algo `mscale_impl` | Sheikh & Bovik, LIVE Lab (UT Austin) | Recherche permissive, citation obligatoire |
| `backends/vif/wavelet_impl.py` | `vif` algo `wavelet_impl` | abhinaukumar/vif | MIT |
| `backends/cw_ssim/wavelet_impl.py` | `cw_ssim` (défaut) | charparr/tundra-snow | MIT |
| `backends/gmsd/numpy_impl.py` | `gmsd` (défaut `numpy`) | bwohlberg/sporco | BSD-3-Clause |
| `backends/pamse/numpy_impl.py` | `pamse` (défaut `numpy`) | bwohlberg/sporco | BSD-3-Clause |
| `backends/ssim/mssim_impl.py` | `msssim` algo `msssim_2` | 4og/mssim | MIT |
| `nr/sharpness/backends/phase/{psi,sharpness_index,lpc_si}.py`, `backends/spectral/s3.py` | `sharpness` (algos `psi`, `sharpness_index`, `lpc_si`, `s3`) | Réimplémentations originales d'Olivier Amram | MIT (auteur du dépôt) |
| `backends/piq/common.py` (dépendance `piq`) | `dists`, `fsim`, `gmsd`, `vif`, `ssim`, `msssim`, `vsi`, `lpips_vgg` (défaut ou algo `piq`) | photosynthesis-team/piq | **Apache-2.0** (vérifié) |
| `backends/{wsnr,reco,niqe,piqe,brisque}/numpy_impl.py`, `nr/focus_fmeasure.py` | `wsnr`, `reco`, `niqe`, `piqe`, `brisque` (défauts), `fmeasure` (27 opérateurs) | Réimplémentations propres, écrites depuis la description publiée de chaque algorithme (voir `REFERENCE_TECHNIQUE.md`) | MIT (auteur du dépôt) |

Les modèles pré-entraînés de NIQE (`backends/niqe/models/modelparameters.mat`)
et BRISQUE (`backends/brisque/models/allmodel`) sont des paramètres
statistiques appris (pas du code) — leur provenance est retracée jusqu'à la
release officielle du LIVE Lab (UT Austin, Mittal/Soundararajan/Bovik pour
NIQE, Mittal/Moorthy/Bovik pour BRISQUE), sous licence de recherche
permissive avec citation obligatoire — voir `models/NOTICE.md` dans chaque
dossier.

### 2.3 Piste écartée : `pyiqa`

`pyiqa` (chaofengc/IQA-PyTorch, https://github.com/chaofengc/IQA-PyTorch) a
été évalué comme remplacement de IQA_pytorch et **écarté** : licence
**PolyForm-Noncommercial-1.0.0**, confirmée sur le fichier LICENSE du
dépôt — usage commercial explicitement interdit (la définition de
« noncommercial » y inclut néanmoins les institutions gouvernementales).
**Pas une licence open-source au sens OSI.** `pyiqa` n'est pas, et n'a
jamais été, une dépendance d'eoqual-metrics — voir `piq` (§2.2) pour le
remplacement effectivement retenu (Apache-2.0). `nlpd` et `mad` restent
sur `IQA_pytorch` (MIT, legacy) faute d'équivalent dans `piq` ; `mad` reste
inutilisable quelle que soit la dépendance installée (voir `config.py`).

### 2.4 Dépendances installées

Toutes BSD/MIT/Apache-2.0 (numpy, scipy, scikit-image, scikit-learn,
opencv-python, sewar, pyrtools, image-similarity-measures, PyWavelets,
joblib, rich, loguru, pandas, IQA_pytorch, piq), à l'exception de `cpbd`
dont le classifieur PyPI indique « Other/Proprietary License » — à vérifier
plus précisément si l'algo passe en utilisation production (voir §3).

## 3. Méthodes non implémentées (roadmap)

Ces métriques ne sont **pas encore portées** dans `eoqual-metrics` — issues
d'une veille bibliographique interne (Hammou et al., "EGB: Image Quality
Assessment Based on Ensemble of Gradient Boosting", CVPRW 2021, et
références qu'il cite). `fsim`, `vsi` et `vif` sont déjà au catalogue (voir
README §Catalogue) ; `mad` y figure aussi mais reste **non fonctionnel** en
l'état, indépendamment de la dépendance installée (§2.3) — pas dupliqué
ci-dessous.

### 3.1 Code source disponible, non intégré

| Méthode | Référence | Raison |
|---|---|---|
| **IFC** (Information Fidelity Criterion) | Sheikh, H. R., Bovik, A. C., de Veciana, G. (2005). "An information fidelity criterion for image quality assessment using natural scene statistics." IEEE TIP, 14(12), 2117-2128. | Précurseur de `vif` (même papier/même équipe, déjà au catalogue) : métrique plus simple, non extraite séparément. Code de référence potentiellement présent dans `pavancm/Visual-Information-Fidelity---Python` (source déjà utilisée pour `vif`, voir §2.2) — à vérifier. |
| **PieAPP** (Perceptual Image-Error Assessment through Pairwise preference) | Prashnani, E., Cai, H., Mostofi, Y., Sen, P. (2018). "PieAPP: Perceptual image-error assessment through pairwise preference." IEEE CVPR, 1808-1817. | Modèle appris (réseau de préférence par paires), dépôt de référence disponible (`prashnani/PerceptualImageError`) mais nécessite un modèle pré-entraîné PyTorch — pas encore intégré. |
| **VSNR** (Visual Signal-to-Noise Ratio) | Chandler, D. M., Hemami, S. S. (2007). "VSNR: A wavelet-based visual signal-to-noise ratio for natural images." IEEE TIP, 16(9), 2284-2298. Voir aussi Farrell, J. et al. (2010), "vSNR and pixel binning" (SPIE). | Code de référence en **MATLAB uniquement** (`sattarab/image-quality-tools`, dossier `metrix_mux/metrix/vsnr`), pas de portage Python identifié à ce jour. À ne pas confondre avec la méthode de destriping du même nom (`vsnr` dans `eoqual-destriping`, algorithme variationnel de suppression de rayures — sans rapport). |

### 3.2 Aucun code source disponible (citation seule)

| Méthode | Référence |
|---|---|
| **EGB** (Ensemble of Gradient Boosting) — *en cours d'étude* | Hammou, D. et al. (2021). "EGB: Image Quality Assessment Based on Ensemble of Gradient Boosting." IEEE CVPRW (NTIRE). |
| **FSIMc** (variante chrominance de FSIM) | Zhang, L., Zhang, L., Mou, X., Zhang, D. (2011). "FSIM: A feature similarity index for image quality assessment." IEEE TIP, 20(8), 2378-2386. Non exposée par les implémentations FSIM actuellement utilisées (voir `fsim` au catalogue, §2.2). |
| **GSM** (Gradient Similarity) | Liu, A., Lin, W., Narwaria, M. (2011). "Image quality assessment based on gradient similarity." IEEE TIP, 21(4), 1500-1512. |
| **MA** | Ma, C., Yang, C.-Y., Yang, X., Yang, M.-H. (2017). "Learning a no-reference quality metric for single-image super-resolution." Computer Vision and Image Understanding, 158, 1-16. |
| **PI** (Perceptual Index, benchmark PIRM) | Blau, Y., Michaeli, T. (2018). "The perception-distortion tradeoff." IEEE CVPR, 6228-6237. |
| **RFSIM** (Riesz-transform Feature SIMilarity) | Zhang, L., Zhang, L., Mou, X. (2010). "RFSIM: A feature based image quality assessment metric using Riesz transforms." IEEE ICIP, 321-324. |
| **SRSIM** (Spectral Residual SIMilarity) | Zhang, L., Li, H. (2012). "SR-SIM: A fast and high performance IQA index based on spectral residual." IEEE ICIP, 1473-1476. |
| **SWD** | Gu, J., Cai, H., Chen, H., Ye, X., Ren, J., Dong, C. (2020). "Image quality assessment for perceptual image restoration: A new dataset, benchmark and metric." arXiv:2011.15002. |
| **WaDIQaM** | Bosse, S., Maniry, D., Müller, K.-R., Wiegand, T., Samek, W. (2017). "Deep neural networks for no-reference and full-reference image quality assessment." IEEE TIP, 27(1), 206-219. |

**Ressources générales** (bancs de comparaison susceptibles de couvrir
plusieurs des métriques ci-dessus, non auditées individuellement) :
`dingkeyan93/IQA-optimization`, `weizhou-geek/Image-Quality-Assessment-Benchmark`.

## 4. Pistes de remédiation restantes (facultatif)

Aucune n'est bloquante pour l'usage standard de la librairie (rien de ce qui
suit n'est atteint par un appel « par défaut ») :

- Les 5 fichiers 🟡 du §2.1 pourraient être réécrits en clean-room (comme
  les métriques du §2.2) ou retirés purement et simplement — ce sont des
  alternatives redondantes (`ssim`/`msssim` ont déjà des algos par défaut
  propres) ; seul `wsnr` perdrait son unique alternative en cas de retrait
  de `metrikz`.
- Vérifier plus précisément la licence PyPI de `cpbd` (§2.4) si son usage
  passe en production.
