# Change Log

## Version History

Non publié

:   -   **Dynamique des images dans les enveloppes
        image-similarity-measures** : `psnr`, `rmse`, `ssim` reçoivent
        `max_p` = dynamique des images (au lieu des 4095 supposés en
        amont : PSNR de 52 dB au lieu de 28 dB sur une image 0-255),
        `fsim` reçoit des images ramenées sur [0, 255] (T2 = 160) et
        `issm` sur [0, 4095] (formule 12 bits : il renvoyait 0 sur toute
        image 8 bits ou flottante). Résultats désormais indépendants de
        l'échelle des valeurs ; `psnr` et `ssim` égaux aux algos `numpy`
        et `skimage`.
    -   **`issm` réimplémenté** (nouvel algo `numpy`, par défaut) d'après
        Aljanabi et al. (2019) : l'algo `image-similarity-measures`
        renvoyait 0 sur toute image 8 bits ou flottante (conception
        12 bits). Choix documentés là où l'article est muet ou
        incohérent (EHS en comptes pour reproduire ses figures, Canny à
        seuils automatiques façon MATLAB, e = nombre d'Euler) ;
        paramètres `data_range` et `ehs`. 1 pour deux images
        identiques, décroissance avec le bruit, < 0.05 entre images sans
        rapport (`tests/test_issm.py`). L'ancien algo reste disponible
        pour comparaison.
    -   **Correction des enveloppes `image-similarity-measures`**
        (`fsim`, `issm`, `sam`, `sre`, `psnr`, `rmse`, `ssim`, `uqi`) :
        l'axe de canal n'est plus ajouté qu'aux images 2D
        (`core.checker.as_channels_last`). En RVB, `sam` mesurait l'angle
        entre des scalaires (≈ 0 au lieu de 3.49° sur l'image de test),
        `ssim`/`uqi` levaient une erreur et `issm` échouait dans OpenCV ;
        les valeurs sur image 2D sont inchangées.
    -   `sam` : `lower_better: True` (c'est un angle), restreint aux
        images multicanal (`ValueError` sur une image 2D, où il valait
        toujours 0), calcul en float64 (≈ 0.006° au lieu de 0 pour deux
        images identiques en float32). Tests
        `tests/test_ism_wrappers.py` (sautés sans image-similarity-measures).
    -   Échec d'import : `logger.catch(reraise=True)` dans `psi`, `lpc_si`,
        `s3`, et `raise` au lieu de `sys.exit(-2)` dans les backends
        `metrikz`, `ssim_2`, `ssim_4` — la trace reste affichée, mais
        l'`ImportError` d'origine est relancée (rattrapable dans un
        notebook ou un programme appelant ; code retour 1 pour un
        script).
    -   **Rejet des arêtes aberrantes** dans `aem`/`sasbem`
        (`detect_and_measure_edges`) : LSF qui s'annule presque (`|Σ LSF|
        / Σ |LSF| < 0.1`, ligne fine) ou MTF > 1.5 avant Nyquist (lobes
        négatifs). Sur l'image `r0`, `aem` passe de 0.167 à 0.050 (4
        arêtes sur 8 rejetées, dont une à MTF@Nyquist = 73). Seuils
        réglables (`max_mtf_overshoot`, `min_edge_sign_ratio`).
    -   `python >= 3.10` (au lieu de 3.12 : aucune construction du code ne
        l'imposait ; vérifié sous 3.10 et 3.11).
    -   `pyfftw` pris sur PyPI (`>= 0.15.1`, au lieu du dépôt git) :
        wheels avec FFTW embarqué (plus de prérequis système), résultats
        identiques ; installé à partir de Python 3.11 (exigence de
        pyfftw), sous 3.10 seules les métriques
        d'image-similarity-measures sont indisponibles.
    -   `cookiecutter` déplacé des dépendances d'exécution vers le groupe
        `dev`.
    -   Intégration continue des tests (`.github/workflows/tests.yml`).
    -   `license.txt` : titulaire des droits (Olivier Amram) au lieu du
        nom du module.
    -   `sharpness_index` réécrit : calcul en float64 (écart relatif ~6e-5 avec
        l'ancienne version, qui travaillait en float32), suppression d'un
        `print` de débogage et de l'arrêt du processus à l'import
        (`logger.catch` + `sys.exit`) ; nouvelle fonction
        `sharpness_stats` (μ, σ, z, TV, SI) ; `pmode` hors de 0..3 lève
        une `ValueError`. Même nom, même paramètre, même défaut.
    -   Premier test automatisé de l'indice
        (`tests/test_sharpness_index.py`) : moments validés par
        Monte-Carlo, décroissance avec le flou et le bruit.
    -   **Dépendance à eoqual-core** (épinglée sur `v0.1.0`) : la
        décomposition périodique et la déquantification de l'indice SI
        (`eoqual.core.fourier`), le binning de l'ESF, la MTF, la
        MTF@Nyquist, la MTF50 et la FWHM d'`aem`/`sasbem`
        (`eoqual.core.edge`), le découpage en tuiles
        (`eoqual.core.tiling`). Scores `aem`/`sasbem`, MTF@Nyquist, MTF50,
        SI et tuiles inchangés (écarts ≤ 1e-9).
    -   **Correction de la FWHM** d'`aem`/`sasbem` (détails
        `return_details`) : calculée sur la LSF lissée d'un demi-pixel,
        par la première traversée du demi-maximum autour du pic.
        L'ancienne valeur (premier/dernier point au-dessus du seuil sur
        tout le profil, sans interpolation) dépassait 15 px sur un rebond
        de bruit ; les nouvelles valeurs sont cohérentes avec le flou
        simulé (ex. ~2.5 px pour une PSF gaussienne de σ = 1 px).

1.1.0 (2026-09-12)

:   -   **Rupture (sans rétrocompatibilité)** : `sharpness(algo="tenengrad")`
        renommé `"tenengrad_otsu"` (et son seuillage par défaut passe de
        `"max_frac"` à `"otsu"`, sans paramètre arbitraire à régler) ;
        `sharpness(algo="brenner")` renommé `"brenner_vertical"`. Motif :
        ces noms entraient en collision avec `fmeasure(algo="TENG"/"BREN")`,
        qui portent le même nom historique mais implémentent une formule
        différente — les scores n'étaient pas comparables malgré la
        ressemblance de nom. Voir `BIBLIOGRAPHY.md` et la docstring de
        `nr/sharpness/__init__.py`.
    -   Clarification (docs + docstrings) : dans `sharpness` et `fmeasure`,
        `algo=` sélectionne une méthode de mesure différente pour chaque
        valeur — contrairement au reste du registre, où `algo=` sélectionne
        une implémentation alternative d'une même définition. Les valeurs
        ne sont pas interchangeables.
    -   Nouvelle métrique FR `jsd` (distance de Jensen-Shannon entre
        histogrammes d'intensité, nombre de classes adaptatif).
    -   Nouvel algo `sharpness(algo="blur_effect")` (Crété et al. 2007, via
        `skimage.measure.blur_effect`), validé contre
        `deepinv.loss.metric.BlurStrength`.
    -   Fix : normalisation `antonel` (Sx/Sy) alignée sur l'équation du
        papier source (moyenne des ratios, pas ratio des moyennes).
    -   Nouveau document `BIBLIOGRAPHY.md` : référence bibliographique de
        chaque méthode nommée, en un seul endroit — `REFERENCE_TECHNIQUE.md`
        et les docstrings n'en gardent qu'un rappel court.
    -   CI : migration du pipeline de documentation de GitLab CI (jamais
        exécuté, dépôt hébergé sur GitHub) vers GitHub Actions + GitHub
        Pages ; fix de génération autoapi (page "API Reference" vide) ;
        fix d'une source Poetry invalide bloquant `poetry install`.

1.0.0 (2026-09-03)

:   -   Première version publique.
    -   **Full-Reference** (34 métriques) : erreur pixel-à-pixel, signal/bruit
        (SNR, PSNR, WSNR), famille SSIM (SSIM, MS-SSIM, CW-SSIM), corrélation,
        perceptuelle (VIF, FSIM, GMSD, DISTS, LPIPS...), spectrale/télédétection
        (ERGAS, SAM, SRE, RASE).
    -   **No-Reference** : BRISQUE, NIQE, PIQE (statistiques apprises), entropie,
        27 opérateurs de mise au point (`fmeasure`), et une façade `sharpness`
        à 16 algorithmes répartis en 5 familles — gradient, phase, spectral,
        perceptuel, et une famille **satellite** calibrée (MTF, ISO 12233 —
        `aem`/`sasbem`, slanted-edge method sur arêtes naturelles).
    -   **Calcul par tuiles** (opt-in) pour toute métrique No-Reference —
        `eoqual.metrics.compute_score_on_tiles`.
    -   Registre central unique (`config.py::METRICS_CONFIGS`) : chaque métrique
        déclare ses algorithmes, son défaut, son sens d'optimalité. Orchestrateur
        `MetricsRunner` pour le calcul par lot, export console (Rich) et CSV.
    -   Dépendances lourdes (`torch`, `piq`) et optionnelles (`svm`) isolées
        dans des extras `pip install "eoqual-metrics[deep|svm]"` — installation
        de base légère.
    -   Tout algorithme obtenu **par défaut** (sans préciser `algo=`) est sous
        licence permissive — voir `THIRD_PARTY_LICENSES.md` pour le détail
        complet et les quelques algorithmes optionnels qui ne le sont pas.
    -   Documentation : `README.md` (installation, catalogue, licence),
        `REFERENCE_TECHNIQUE.md` (méthode/implémentation/limites par famille),
        `THIRD_PARTY_LICENSES.md` (audit des licences tierces),
        `CONTRIBUTING.md` (signaler un bug, proposer/implémenter une métrique).
        Génération automatique via Sphinx + autoapi (pipeline CI).
