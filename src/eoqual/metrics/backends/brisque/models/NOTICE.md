# Provenance — `allmodel`

Modèle SVR (epsilon-SVR, noyau RBF, 770 vecteurs de support, 36
dimensions) utilisé par `brisque`. Format texte standard libsvm — mêmes
valeurs numériques que celles distribuées avec la release officielle
BRISQUE (Mittal, Moorthy, Bovik) sur `live.ece.utexas.edu` : entraîné sur
la base LIVE (images pristines + dégradées notées par des sujets humains).

**Source** : Anish Mittal, Anush K. Moorthy, Alan C. Bovik — Laboratory
for Image and Video Engineering (LIVE), University of Texas at Austin.
Release officielle : `http://live.ece.utexas.edu/research/quality/BRISQUE_release.zip`.

**Licence** (citée sur la page LIVE) : utilisation, copie, modification et
distribution libres, à condition que la notice de copyright soit
conservée et que la source (LIVE, UT Austin) soit reconnue dans toute
publication utilisant ce code/modèle, avec citation de :

> A. Mittal, A. K. Moorthy and A. C. Bovik, "No-Reference Image Quality
> Assessment in the Spatial Domain," IEEE Transactions on Image
> Processing, 2012.

Ce fichier est conservé tel quel (voir `THIRD_PARTY_LICENSES.md`) : c'est
un modèle statistique appris, pas du code — il ne peut pas être
"réimplémenté", seulement réutilisé sous sa licence d'origine (permissive,
ci-dessus). La prédiction SVR elle-même est recalculée en numpy pur (voir
`numpy_impl.py`), sans dépendance à libsvm.
