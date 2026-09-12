# Change Log

## Version History

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
