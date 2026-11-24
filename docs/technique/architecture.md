# Architecture générale

## 1. Architecture générale

Un unique registre, `config.py::METRICS_CONFIGS`, décrit chaque métrique :
ses algorithmes disponibles (`algo`), l'algorithme par défaut (`default`),
le mode (`FR`/`NR`), le sens d'optimalité (`lower_better`), et la
restriction éventuelle de bande (`only: 'GRAY'|'RGB'`). Trois façons de
consommer ce registre :

- **Appel direct** : `eoqual.metrics.<nom_métrique>(GT, P, algo=...)` (FR)
  ou `(P, algo=...)` (NR).
- **Façade multi-algo** : certaines familles (`sharpness`) exposent une
  seule fonction publique qui dispatché en interne vers un backend selon
  `algo=`.
- **Orchestrateur `MetricsRunner`** (`runner.py`) : construit un plan
  d'exécution depuis le registre (ou un sous-ensemble choisi), calcule sur
  une ou plusieurs paires, accumule, exporte (`export/console.py`,
  `export/csv.py`).

Convention de code : chaque métrique/algo vit dans son propre module sous
`backends/<famille>/<impl>.py` (FR) ou
`nr/sharpness/backends/<famille>/<algo>.py` (NR sharpness) ; la fonction
publique de plus haut niveau (`fr/*.py`, `nr/*.py`) ne fait que
sélectionner le bon backend et normaliser les entrées/sorties
(`core/checker.py::initial_check`, `core/image_utils.py`).
