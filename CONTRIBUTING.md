# Contribuer à eoqual-metrics

Les contributions sont bienvenues, sous toutes leurs formes : signalement
de bug, correction, proposition de nouvelle métrique, implémentation d'une
métrique déjà proposée, amélioration de la documentation.

## Signaler un bug

Ouvrez une issue avec :
- la métrique et l'`algo=` concernés (`m.list_metrics()` pour voir le
  registre complet dans votre environnement) ;
- un exemple minimal reproduisant le problème (image de test, code) ;
- le résultat obtenu vs. attendu, et si pertinent la trace complète
  (le runner ne masque volontairement aucune exception — voir
  `REFERENCE_TECHNIQUE.md` §5.1).

## Proposer une métrique

Ouvrez une issue décrivant la métrique (nom, référence bibliographique si
elle existe, ce qu'elle apporte par rapport aux métriques déjà cataloguées
— voir `REFERENCE_TECHNIQUE.md` pour l'existant). Pas besoin de savoir
l'implémenter pour la proposer.

## Implémenter une métrique ou un algorithme

Convention du projet — un algorithme = une fonction `ndarray -> float`
(ou `-> (float, dict)` si un détail/une carte a du sens), placée dans son
propre module :

- **Full-Reference** (`f(GT, P, ...)`) : `src/eoqual/metrics/backends/<famille>/<impl>.py`,
  dispatché depuis `fr/<famille>.py`.
- **No-Reference** (`f(P, ...)`) : `src/eoqual/metrics/nr/sharpness/backends/<famille>/<algo>.py`
  pour la façade `sharpness`, ou `src/eoqual/metrics/backends/<métrique>/<impl>.py`
  pour les autres (`brisque`, `niqe`, `piqe`...).

Puis déclarer l'algorithme dans `config.py::METRICS_CONFIGS` (nom, mode
FR/NR, restriction de bande éventuelle, sens d'optimalité `lower_better`)
et l'ajouter au dispatch de la fonction publique correspondante.

**Licence** — condition de fusion, pas une formalité : toute
implémentation reprenant ou s'inspirant de près d'un code publié ailleurs
doit citer sa source et la licence de cette source dans le module (voir
les en-têtes existants pour le format). Une implémentation dont la licence
d'origine n'est pas identifiée, ou qui s'avère copyleft/non permissive,
**ne sera pas acceptée en algorithme par défaut** d'une métrique — voir
`THIRD_PARTY_LICENSES.md` pour le principe déjà appliqué à l'existant, et
son avertissement en tête : une erreur de licence constatée après coup
sera corrigée, implémentation retirée si nécessaire. Éviter d'y exposer le
projet en amont.

## Tests

Le projet n'a pas encore de suite `pytest` assertive (voir
`REFERENCE_TECHNIQUE.md` §5.4) — en attendant, toute nouvelle
implémentation gagne à être accompagnée d'un script dans `examples/`
démontrant son usage, et idéalement d'une comparaison numérique contre une
référence connue (valeur publiée, implémentation de référence) dans la
description de la pull request.

## Documentation

Toute métrique/algo nouveau ou modifié doit rester reflété dans
`README.md` (catalogue) et `REFERENCE_TECHNIQUE.md` (méthode,
implémentation, limites) — les docstrings seules ne suffisent pas, ces
deux documents sont la carte d'ensemble du projet.
