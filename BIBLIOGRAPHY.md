# Bibliographie — eoqual-metrics

Référence bibliographique de chaque métrique/algorithme nommé du
catalogue, en un seul endroit. Les métriques génériques sans publication
unique identifiable (`mae`, `mse`, `rmse`, `psnr`, `ncc`, `ergas`, `sam`,
`rase`, `sre`...) ne sont pas listées — ce sont des formules standard de
traitement du signal / télédétection, pas des méthodes attribuables à un
article précis.

Ce document est la source de vérité pour les citations complètes ;
`REFERENCE_TECHNIQUE.md` et les docstrings n'en gardent qu'un rappel
court (auteur, année). Les docstrings de chaque module restent la source
technique de référence (formule exacte, paramètres).

> Certaines entrées sont marquées **réf. incomplète** : la citation
> précise (volume, pages, DOI) n'a pas pu être établie avec confiance à
> partir du code ou d'une source vérifiée — seuls le titre/auteurs/année
> connus sont indiqués plutôt que d'inventer des détails. Signaler via
> `CONTRIBUTING.md` si vous avez la référence exacte.

## Full-Reference (FR)

### Signal-based

| Métrique | Référence |
|---|---|
| `wsnr` (pondération CSF) | Mannos, J. & Sakrison, D. (1974). *The effects of a visual fidelity criterion on the encoding of images*. IEEE Trans. Information Theory, 20(4), 525-536. |

### Famille SSIM

| Métrique | Référence |
|---|---|
| `ssim`, `msssim` | Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004). *Image quality assessment: From error visibility to structural similarity*. IEEE Transactions on Image Processing, 13(4), 600-612. |
| `cw_ssim` | voir `backends/cw_ssim/wavelet_impl.py` — variante en ondelettes complexes de Wang et al. (2004), pas de publication séparée identifiée pour cette variante spécifique (**réf. incomplète**). |

### Corrélation / statistique

| Métrique | Référence |
|---|---|
| `jsd` | Lin, J. (1991). *Divergence measures based on the Shannon entropy*. IEEE Transactions on Information Theory, 37(1), 145-151. (divergence de Jensen-Shannon elle-même ; `jsd` ici est une implémentation clean-room, voir docstring de `fr/correlation.py::jsd`) |

### Perceptuelle

| Métrique | Référence |
|---|---|
| `vif` | Sheikh, H. R., & Bovik, A. C. (2006). *Image information and visual quality*. IEEE Transactions on Image Processing, 15(2), 430-444. |
| `gmsd` | Xue, W., Zhang, L., Mou, X., & Bovik, A. C. (2014). *Gradient magnitude similarity deviation: A highly efficient perceptual image quality index*. IEEE Transactions on Image Processing, 23(2), 684-695. |
| `pamse` | Xue, W., Zhang, L., & Mou, X. (2013). *Perceptual fidelity aware mean squared error*. IEEE International Conference on Computer Vision (ICCV). |
| `reco` | Baroncini, V., Capodiferro, L., Di Claudio, E. D., & Jacovitti, G. (2009). *The polar edge coherence: a quasi blind metric for video quality assessment*. EUSIPCO 2009, Glasgow, 564-568. |
| `dists` | Ding, K., Ma, K., Wang, S., & Simoncelli, E. P. — *Image Quality Assessment: Unifying Structure and Texture Similarity*. Voir https://github.com/dingkeyan93/DISTS (**réf. incomplète** — année/venue non capturées dans le code). |
| `lpips_vgg` | Zhang, R., Isola, P., Efros, A. A., Shechtman, E., & Wang, O. — *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric*. Voir https://github.com/richzhang/PerceptualSimilarity (**réf. incomplète**). |
| `vsi` | voir https://ieeexplore.ieee.org/document/6873260 (**réf. incomplète** — titre/auteurs non capturés dans le code). |
| `fsim` | Non documentée dans le code (**réf. incomplète**) — méthode classique de la littérature IQA (congruence de phase + gradient), auteurs Zhang, Zhang, Mou & Zhang. |
| `nlpd` | voir https://www.cns.nyu.edu/~lcv/NLPyr/ (**réf. incomplète**). |
| `mad` | Non documentée dans le code (**réf. incomplète**). |
| `issm` | Non documentée dans le code au-delà de `image-similarity-measures` (**réf. incomplète**). |

## No-Reference (NR)

### Blind IQA

| Métrique | Référence |
|---|---|
| `niqe` | Mittal, A., Soundararajan, R., & Bovik, A. C. (2013). *Making a "Completely Blind" Image Quality Analyzer*. IEEE Signal Processing Letters, 20(3), 209-212. |
| `brisque` | Mittal, A., Moorthy, A. K., & Bovik, A. C. (2012). *No-Reference Image Quality Assessment in the Spatial Domain*. IEEE Transactions on Image Processing, 21(12), 4695-4708. |
| `piqe` | Venkatanath, N., Praneeth, D., Bh, M. C., Channappayya, S. S., & Medasani, S. S. (2015). *Blind Image Quality Evaluation Using Perception Based Features*. National Conference on Communications (NCC), IEEE. |

### Focus / fmeasure

Référence de synthèse pour les 27 opérateurs : Pertuz, S., Puig, D., &
Garcia, M. A. (2013). *Analysis of focus measure operators for
shape-from-focus*. Pattern Recognition, 46(5), 1415-1432.
DOI: 10.1016/j.patcog.2012.11.011

| Code | Référence individuelle |
|---|---|
| `ACMO` | Shirvaikar, M. V. (2004). |
| `BREN` | Santos, A., et al. (1997). — extension bidirectionnelle de Brenner (1976), voir `sharpness(algo="brenner_vertical")` ci-dessous pour l'original |
| `CONT` | Nanda, P. K. & Cutler, R. (2001). |
| `CURV` | Helmli, F. S. & Scherer, S. (2001). |
| `DCTE` | Shen, C.-H. & Chen, H. H. (2006). |
| `DCTR` | Lee, S. Y., et al. (2009). |
| `GDER` | Geusebroek, J.-M., et al. (2000). |
| `GLLV` | Pech-Pacheco, J. L., et al. (2000). |
| `GLVA` | Krotkov, E. (1986). |
| `GLVN` | Santos, A., et al. (1997). |
| `GRAE` | Subbarao, M., et al. (1992). |
| `GRAS` | Eskicioglu, A. M. & Fisher, P. S. (1995). |
| `GRAT` | Santos, A., et al. (1997). |
| `HELM` | Helmli, F. S. & Scherer, S. (2001). |
| `HISR` | Firestone, L., et al. (1991). |
| `LAPD` | Thelen, A., et al. (2009). |
| `LAPE` | Subbarao, M., et al. (1992). |
| `LAPM` | Nayar, S. K. & Nakagawa, Y. (1989). |
| `LAPV` | Pech-Pacheco, J. L., et al. (2000). *Diatom autofocusing in brightfield microscopy*. ICPR 2000. — noyau Pertuz/MATLAB, voir `sharpness(algo="laplacian")` pour la variante à noyau OpenCV |
| `SFIL` | Minhas, R., et al. (2009). |
| `SFRQ` | Eskicioglu, A. M. & Fisher, P. S. (1995). |
| `TENG` | Krotkov, E. (1986). — voir `sharpness(algo="tenengrad_otsu")` pour la variante seuillée |
| `TENV` | Pech-Pacheco, J. L., et al. (2000). |
| `VOLA` | Santos, A., et al. (1997). |
| `WAVR` | Yang, G. & Nelson, B. J. (2003). |
| `WAVS` | Yang, G. & Nelson, B. J. (2003). |
| `WAVV` | Yang, G. & Nelson, B. J. (2003). |

Ces 27 citations individuelles (années précises, sans détail de venue)
sont reprises telles quelles de la table de correspondance de Pertuz et
al. (2013) ci-dessus, elle-même la source primaire pour l'attribution de
chaque opérateur — **réf. incomplète** au sens du venue/pages exacts de
chaque publication d'origine, mais l'attribution auteur/année est fiable.

### Sharpness

| Algo | Référence |
|---|---|
| `tenengrad_otsu` | Krotkov, E. (1987). *Focusing*. Int J Comput Vision 1, 223-237. https://doi.org/10.1007/BF00127822 — voir `fmeasure(algo="TENG")` pour la variante sans seuillage |
| `laplacian` | Pech-Pacheco, J. L., et al. (2000). *Diatom autofocusing in brightfield microscopy*. ICPR 2000. — noyau OpenCV, voir `fmeasure(algo="LAPV")` pour la variante à noyau Pertuz/MATLAB |
| `brenner_vertical` | Brenner, J. F., et al. (1976). *An automated microscope for cytologic research: A preliminary evaluation*. The Journal of Histochemistry and Cytochemistry. — voir `fmeasure(algo="BREN")` pour l'extension bidirectionnelle |
| `lpc_si` | Hassen, R., Wang, Z., & Salama, M. M. A. — *Image sharpness assessment based on local phase coherence*. IEEE Transactions on Image Processing (**réf. incomplète** — année/volume non capturés dans le code, ~2013). |
| `sharpness_index` | Blanchet, G., Moisan, L., & Rougé, B. *Measuring the global phase coherence of an image* (ICIP 2008) ; Blanchet, G. & Moisan, L. *An explicit Sharpness Index related to Global Phase Coherence* (ICASSP 2012) ; Leclaire, A. & Moisan, L. *No-reference image quality assessment and blind deblurring with sharpness metrics exploiting Fourier phase information* (Journal of Mathematical Imaging and Vision, 2015). |
| `psi` | Feichtenhofer, C., Fassold, H., & Schallauer, P. (2013). *A perceptual image sharpness metric based on local edge gradient analysis*. IEEE Signal Processing Letters, 20(4), 379-382. |
| `s3` | Vu, C. T., Phan, T. D., & Chandler, D. M. — *S3: A spectral and spatial measure of local perceived sharpness in natural images*. IEEE Transactions on Image Processing (**réf. incomplète** — année/volume non capturés dans le code, ~2012). |
| `mtf`, `aem`, `sasbem` | ISO 12233 (slanted-edge method) — norme, pas un article de recherche unique. |
| `cpbd` | Narvekar, N. D., & Karam, L. J. (2011). *A no-reference image blur metric based on the cumulative probability of blur detection*. IEEE Transactions on Image Processing. |
| `blur_effect` | Crété, F., Dolmiere, T., Ladret, P., & Nicolas, M. (2007). *The Blur Effect: Perception and Estimation with a New No-Reference Perceptual Blur Metric*. Proc. SPIE 6492, Human Vision and Electronic Imaging XII, 64920I. https://hal.archives-ouvertes.fr/hal-00232709 — implémenté via `skimage.measure.blur_effect`. |
| `antonel` | Antonel, L. G. (2024). *A Novel No-Reference Image Quality Metric for Assessing Sharpness in Satellite Imagery*. arXiv:2410.10488. |
| `blur_kernel` | Anger, J., de Franchis, C., & Facciolo, G. (2019). *Assessing the Sharpness of Satellite Images: Study of the PlanetScope Constellation*. IGARSS 2019. — principe de scoring repris, pas la méthode d'estimation du noyau (voir docstring). |
