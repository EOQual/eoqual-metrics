"""
Export console et HTML des résultats de métriques.

Ce module fournit :func:`display_table`, une fonction autonome qui
affiche les résultats accumulés sous forme de tableau Rich coloré.
Il ne dépend ni de :class:`~eoqual.metrics.runner.MetricsRunner` ni
d'aucun calcul — il reçoit uniquement des données.

L'affichage met en évidence la valeur optimale de chaque métrique
(en vert) selon le sens d'optimalité défini dans
:data:`~eoqual.metrics.config.METRICS_CONFIGS`.

Exemple
-------
>>> from eoqual.metrics.runner import MetricsRunner
>>> from eoqual.metrics.export.console import display_table
>>> from eoqual.metrics.config import METRICS_CONFIGS
>>>
>>> runner = MetricsRunner({'psnr': 'numpy', 'ssim': 'skimage'})
>>> for gt, p in pairs:
...     runner.compute(gt, p)
...     runner.accumulate()
>>> display_table(runner.results_accumulated, METRICS_CONFIGS)
"""

from __future__ import annotations

import tempfile
import webbrowser
from collections import OrderedDict
from typing import List, Optional, Union

import numpy as np
from loguru import logger
from rich.console import Console
from rich.table import Table

# Type alias (miroir de runner.py pour éviter l'import circulaire)
MetricAccumulated = dict


def _optimal_values(
    accumulated: List[MetricAccumulated],
    configs: OrderedDict,
) -> dict:
    """
    Calcule la valeur optimale de chaque métrique selon ``lower_better``.

    Parameters
    ----------
    accumulated : List[MetricAccumulated]
        Historique des valeurs (sortie de :meth:`~MetricsRunner.accumulate`).
    configs : OrderedDict
        Registre des métriques (:data:`~eoqual.metrics.config.METRICS_CONFIGS`).

    Returns
    -------
    dict
        ``{nom_métrique: valeur_optimale}``

    Raises
    ------
    ValueError
        Si ``lower_better`` n'est pas défini (``None``) pour une métrique
        présente dans les résultats.
    """
    optimal: dict = {}
    for entry in accumulated:
        name = entry["metric"]
        lower_better = configs[name]["lower_better"]
        values = np.array(
            [v for v in entry["value"] if v is not None], dtype=float
        )
        if values.size == 0:
            continue
        if lower_better is True:
            optimal[name] = float(np.min(values))
        elif lower_better is False:
            optimal[name] = float(np.max(values))
        else:
            raise ValueError(
                f"lower_better non défini pour la métrique '{name}'. "
                "Définissez-le dans METRICS_CONFIGS avant d'afficher."
            )
    return optimal


def _format_value(
    value: Optional[float],
    optimal: float,
    precision: int,
) -> str:
    """
    Formate une valeur pour Rich, en la colorant si elle est optimale.

    Parameters
    ----------
    value : float or None
        Valeur à formater.
    optimal : float
        Valeur optimale de référence pour cette métrique.
    precision : int
        Nombre de décimales.

    Returns
    -------
    str
        Chaîne prête pour Rich (avec balises de couleur si optimal).
    """
    if value is None:
        return "—"
    formatted = f"{value:.{precision}f}"
    if value == optimal:
        return f"[bold green]{formatted}[/bold green]"
    return formatted


def display_table(
    accumulated: List[MetricAccumulated],
    configs: OrderedDict,
    titles: Optional[List[str]] = None,
    orientation: str = "horizontal",
    legend: Optional[str] = None,
    output: str = "both",
    precision: int = 4,
) -> None:
    """
    Affiche les résultats accumulés dans un tableau Rich.

    Le tableau met en évidence la valeur optimale de chaque métrique
    en vert. Les métriques dont toutes les valeurs sont ``None`` sont
    automatiquement exclues de l'affichage.

    Parameters
    ----------
    accumulated : List[MetricAccumulated]
        Historique des valeurs, tel que retourné par
        :meth:`~eoqual.metrics.runner.MetricsRunner.accumulate`.
    configs : OrderedDict
        Registre des métriques (:data:`~eoqual.metrics.config.METRICS_CONFIGS`).
    titles : List[str] or None, optional
        Titres des colonnes images. Si ``None``, les colonnes sont nommées
        ``Img_1``, ``Img_2``, etc. Par défaut ``None``.
    orientation : {'horizontal', 'vertical'}, optional
        * ``"horizontal"`` (défaut) — une ligne par métrique, une colonne par image.
        * ``"vertical"`` — une ligne par image, une colonne par métrique.
    legend : str or None, optional
        Titre du tableau. Par défaut ``None``.
    output : {'console', 'html', 'both'}, optional
        Destination de l'affichage :

        * ``"console"`` — terminal uniquement.
        * ``"html"`` — fichier HTML temporaire ouvert dans le navigateur.
        * ``"both"`` (défaut) — les deux.
    precision : int, optional
        Nombre de décimales pour les valeurs numériques. Par défaut ``4``.

    Raises
    ------
    ValueError
        Si ``titles`` ne correspond pas au nombre d'images,
        si ``orientation`` ou ``output`` sont invalides,
        ou si ``lower_better`` est ``None`` pour une métrique affichée.

    Examples
    --------
    >>> display_table(
    ...     runner.results_accumulated,
    ...     METRICS_CONFIGS,
    ...     titles=["Original", "Compressed", "Reconstructed"],
    ...     orientation="vertical",
    ...     legend="Comparaison de compression",
    ... )
    """
    # --- Validation ---
    if orientation not in {"horizontal", "vertical"}:
        raise ValueError(
            f"'orientation' doit être 'horizontal' ou 'vertical', reçu : {orientation!r}"
        )
    if output not in {"console", "html", "both"}:
        raise ValueError(
            f"'output' doit être 'console', 'html' ou 'both', reçu : {output!r}"
        )

    # Filtrage des métriques sans aucune valeur valide
    valid = [e for e in accumulated if any(v is not None for v in e["value"])]
    if not valid:
        logger.warning("display_table : aucun résultat valide à afficher.")
        return

    n_images = len(valid[0]["value"])
    if titles is not None and len(titles) != n_images:
        raise ValueError(
            f"'titles' contient {len(titles)} élément(s), "
            f"mais il y a {n_images} image(s)."
        )

    col_names = titles if titles else [f"Img_{i + 1}" for i in range(n_images)]
    optimal = _optimal_values(valid, configs)

    # --- Construction du tableau Rich ---
    console = Console(record=True)
    table = Table(title=legend, show_header=True, header_style="bold magenta")

    if orientation == "horizontal":
        table.add_column("Métrique", style="bold magenta", no_wrap=True)
        table.add_column("Algo", style="dim")
        for col in col_names:
            table.add_column(col, justify="right")

        for entry in valid:
            name = entry["metric"]
            opt = optimal.get(name)
            row = [name, entry["algo"]]
            for value in entry["value"]:
                row.append(
                    _format_value(value, opt, precision)
                    if opt is not None
                    else ("—" if value is None else f"{value:.{precision}f}")
                )
            table.add_row(*row)

    else:  # vertical
        metric_names = [e["metric"] for e in valid]
        table.add_column("", style="bold magenta", no_wrap=True)
        for name in metric_names:
            table.add_column(name, justify="right")

        for i, col in enumerate(col_names):
            row = [col]
            for entry in valid:
                name = entry["metric"]
                value = entry["value"][i]
                opt = optimal.get(name)
                row.append(
                    _format_value(value, opt, precision)
                    if opt is not None
                    else ("—" if value is None else f"{value:.{precision}f}")
                )
            table.add_row(*row)

    # --- Rendu ---
    if output in {"console", "both"}:
        console.print()
        console.print(table)
        console.print()

    if output in {"html", "both"}:
        with tempfile.NamedTemporaryFile(
            mode="w", delete=False, suffix=".html"
        ) as f:
            console.save_html(f.name)
            webbrowser.open(f"file://{f.name}")
