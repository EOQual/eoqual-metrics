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

## 3. Pistes de remédiation restantes (facultatif)

Aucune n'est bloquante pour l'usage standard de la librairie (rien de ce qui
suit n'est atteint par un appel « par défaut ») :

- Les 5 fichiers 🟡 du §2.1 pourraient être réécrits en clean-room (comme
  les métriques du §2.2) ou retirés purement et simplement — ce sont des
  alternatives redondantes (`ssim`/`msssim` ont déjà des algos par défaut
  propres) ; seul `wsnr` perdrait son unique alternative en cas de retrait
  de `metrikz`.
- Vérifier plus précisément la licence PyPI de `cpbd` (§2.4) si son usage
  passe en production.
