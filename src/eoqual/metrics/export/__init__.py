"""
Modules d'export des résultats de métriques.

Chaque sous-module est indépendant et ne connaît pas le runner —
il reçoit uniquement ``results_accumulated`` et ``METRICS_CONFIGS``.

Modules disponibles
-------------------
* :mod:`~eoqual.metrics.export.console` — affichage Rich (terminal + HTML)
* :mod:`~eoqual.metrics.export.csv` — export fichier CSV
"""
