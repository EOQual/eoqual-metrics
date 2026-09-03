# Change Log

## Version History

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
