# Formules — eoqual-metrics

Formule mathématique exacte de chaque métrique consolidée en une seule
implémentation (`numpy`), avec un glossaire de notation commun. Portée
volontairement limitée à ces métriques pour l'instant, plutôt que
d'essayer de couvrir les 41 métriques du registre d'un coup et de risquer
de ne pas pouvoir tenir ce document à jour — voir `REFERENCE_TECHNIQUE.md`
pour la méthode de chaque famille, `BIBLIOGRAPHY.md` pour les références
bibliographiques (ce document ne les répète pas).

## Glossaire de notation

| Symbole | Signification |
|---|---|
| `GT` | Image de référence (*ground truth*), tableau 2D `(H, W)` ou 3D `(H, W, C)` |
| `P` | Image traitée / dégradée, même forme que `GT` |
| `N` | Nombre total de pixels de `GT` (= `H × W`, ou `H × W × C` si multicanal) |
| `i` | Indice de pixel, `i = 1...N` |
| `GT_i`, `P_i` | Valeur du pixel `i` dans `GT`/`P` |
| `μ(X)` | Moyenne arithmétique des valeurs de `X` |
| `σ(X)` | Écart-type des valeurs de `X` (population, pas d'estimateur non biaisé) |
| `R` | Dynamique de `GT` : `R = max(GT) - min(GT)` |
| `‖X‖` | Norme euclidienne (L2) du vecteur `X` à plat |
| `c` | Indice de canal (R, G, B) pour `psnrc` |

Toutes les implémentations numpy travaillent en `float64`, quelle que soit
la précision d'entrée.

## MSE — Mean Squared Error

```
MSE(GT, P) = (1/N) · Σ_i (GT_i - P_i)²
```

`np.mean((GT - P) ** 2)`. Plus faible = meilleure qualité. Pas de
référence bibliographique unique (définition standard).

## RMSE — Root Mean Squared Error

```
RMSE(GT, P) = √MSE(GT, P)
```

`np.sqrt(np.mean((GT - P) ** 2))`. Unités de pixel (pas normalisée).
Plus faible = meilleure qualité.

## PSNR — Peak Signal-to-Noise Ratio

```
PSNR(GT, P) = 10 · log10( R² / MSE(GT, P) )
```

avec `R = max(GT) - min(GT)` (dynamique de la référence, pas une valeur
fixe supposée comme 255 ou 4095). En décibels (dB). Plus élevé = meilleure
qualité. `+∞` si `GT == P`.

## PSNR-C — PSNR couleur

```
PSNR-C(GT, P) = (1/3) · Σ_{c ∈ {R,G,B}} PSNR(GT_c, P_c)
```

Moyenne arithmétique du PSNR calculé indépendamment sur chaque canal.
Nécessite des images RGB `(H, W, 3)` — retourne `None` sinon (avec un
avertissement). Même convention de dynamique que `PSNR` (par canal).

## NCC — Normalized Cross-Correlation (coefficient de Pearson)

```
NCC(GT, P) = μ( (GT - μ(GT)) · (P - μ(P)) ) / ( σ(GT) · σ(P) )
```

Dans `[-1, 1]` ; `1` = corrélation parfaite. `0` si l'une des deux images
est de variance nulle (évite une division par zéro). Plus élevé =
meilleure qualité.

## NDP — Normalized Dot Product (angle cosinusoïdal / VCAD)

```
NDP(GT, P) = ( GT · P ) / ( ‖GT‖ · ‖P‖ )
```

Produit scalaire des deux images aplaties en vecteurs, normalisé par le
produit de leurs normes — c'est le cosinus de l'angle entre les deux
vecteurs. Dans `[-1, 1]` : `1` = vecteurs colinéaires positifs, `0` =
orthogonaux, `-1` = opposés. Plus élevé = meilleure qualité.

---

*Les six métriques ci-dessus n'acceptent plus qu'un seul algorithme
(`"numpy"`) depuis l'audit documenté dans `REFERENCE_TECHNIQUE.md` §2.1/
§2.2/§2.4 : les anciennes alternatives (bibliothèques tierces) donnaient
un résultat statistiquement identique pour un temps de calcul équivalent
— les garder n'apportait rien, sauf pour l'algo `image-similarity-measures`
de `rmse`, qui calculait en réalité une métrique différente (voir §2.1).*
