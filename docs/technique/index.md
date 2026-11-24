# Documents techniques

Documents de traçabilité de l'ensemble des métriques cataloguées dans
`eoqual-metrics`. Corps principal centré sur l'**état actuel** : **méthode**
(que calcule-t-on, indépendamment du code) → **implémentation** (où c'est
codé, avec quoi, quel est le défaut) → **limites connues**. Les pistes
(non engagées) sont regroupées au [§6](limites.md#6-pistes) ; ce qui change d'une version à
l'autre est dans `CHANGELOG.md`.

Ces documents ne remplacent pas les docstrings (référence API précise,
autoapi/Sphinx), [bibliographie](bibliographie.md) (référence bibliographique complète de
chaque méthode nommée — ces documents ne gardent qu'un rappel court
auteur/année) ni `THIRD_PARTY_LICENSES.md` (audit des licences tierces,
document séparé) : il sert de carte d'ensemble pour comprendre *pourquoi*
chaque famille existe et *où* la retrouver — à maintenir à chaque
évolution notable du registre `config.py`.

Les chapitres gardent une numérotation unique d'un document à l'autre : un
renvoi « §n » désigne toujours le même chapitre.

| Document | Chapitres | Contenu |
|---|---|---|
| [Architecture générale](architecture.md) | 1 | Registre, familles, runner, conventions communes |
| [Full-Reference : comparer deux images](full-reference.md) | 2 | Erreur pixel, SNR/PSNR, SSIM, corrélation, perceptuelles, spectrales |
| [No-Reference : évaluer une image seule](no-reference.md) | 3 | IQA aveugle, focus et entropie, netteté (5 sous-familles, dont MTF sur arêtes naturelles), calcul par tuiles |
| [Reduced-Reference, limites transverses et pistes](limites.md) | 4, 5, 6 | Reduced-Reference, gestion des erreurs, licences, dépendances, tests, pistes |
| [Formulaire](formulaire.md) | | Formules des métriques |
| [Bibliographie](bibliographie.md) | | Références des métriques |

```{toctree}
:hidden:
:maxdepth: 2

architecture
full-reference
no-reference
limites
formulaire
bibliographie
```
