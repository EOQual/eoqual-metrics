"""
Export CSV des résultats de métriques.

Ce module fournit :func:`save_csv`, une fonction autonome qui écrit
les résultats accumulés dans un fichier CSV structuré.

Format du CSV produit
---------------------
Une ligne d'en-tête, puis une ligne par couple ``(métrique, algo)`` :

.. code-block:: text

    Metric,Algo,Img_1,Img_2,Img_3
    psnr,numpy,34.1200,32.8700,35.4400
    ssim,skimage,0.9412,0.9103,0.9521

Exemple
-------
>>> from eoqual.metrics.export.csv import save_csv
>>> save_csv(runner.results_accumulated, "results.csv", titles=["REF", "A", "B"])
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger

MetricAccumulated = dict


def save_csv(
    accumulated: List[MetricAccumulated],
    path: Union[str, Path],
    titles: Optional[List[str]] = None,
    precision: int = 4,
    delimiter: str = ",",
) -> Path:
    """
    Écrit les résultats accumulés dans un fichier CSV.

    Chaque ligne correspond à un couple ``(métrique, algo)``.
    Les colonnes suivantes contiennent les valeurs pour chaque image,
    dans l'ordre d'accumulation.

    Parameters
    ----------
    accumulated : List[MetricAccumulated]
        Historique des valeurs, tel que retourné par
        :meth:`~eoqual.metrics.runner.MetricsRunner.accumulate`.
    path : str or Path
        Chemin du fichier CSV à créer. Les répertoires parents sont
        créés automatiquement s'ils n'existent pas.
    titles : List[str] or None, optional
        Titres des colonnes images. Si ``None``, les colonnes sont nommées
        ``Img_1``, ``Img_2``, etc. Par défaut ``None``.
    precision : int, optional
        Nombre de décimales pour les valeurs numériques. Par défaut ``4``.
        Ignoré pour les valeurs ``None`` (écrites comme chaîne vide).
    delimiter : str, optional
        Délimiteur de colonnes. Par défaut ``","`` (CSV standard).
        Utiliser ``";"`` pour Excel en locale française.

    Returns
    -------
    Path
        Chemin absolu du fichier écrit.

    Raises
    ------
    ValueError
        Si ``titles`` ne correspond pas au nombre d'images dans ``accumulated``.
    OSError
        Si le fichier ne peut pas être créé (permissions, espace disque...).

    Examples
    --------
    Enregistrement simple :

    >>> save_csv(runner.results_accumulated, "results.csv")

    Avec titres et délimiteur point-virgule (Excel FR) :

    >>> save_csv(
    ...     runner.results_accumulated,
    ...     "results.csv",
    ...     titles=["Original", "Compressé", "Reconstruit"],
    ...     delimiter=";",
    ... )
    """
    if not accumulated:
        logger.warning("save_csv : aucune donnée à écrire.")
        return Path(path).resolve()

    n_images = len(accumulated[0]["value"])
    if titles is not None and len(titles) != n_images:
        raise ValueError(
            f"'titles' contient {len(titles)} élément(s), "
            f"mais il y a {n_images} image(s)."
        )

    col_names = titles if titles else [f"Img_{i + 1}" for i in range(n_images)]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile, delimiter=delimiter)

        # En-tête
        writer.writerow(["Metric", "Algo", *col_names])

        # Données
        for entry in accumulated:
            values_str = [
                f"{v:.{precision}f}" if v is not None else ""
                for v in entry["value"]
            ]
            writer.writerow([entry["metric"], entry["algo"], *values_str])

    logger.success(f"Résultats sauvegardés dans : {output_path.resolve()}")
    return output_path.resolve()


def load_csv(
    path: Union[str, Path],
    delimiter: str = ",",
) -> List[MetricAccumulated]:
    """
    Relit un fichier CSV produit par :func:`save_csv`.

    Permet de recharger des résultats précédemment sauvegardés pour
    les réafficher ou les comparer sans recalcul.

    Parameters
    ----------
    path : str or Path
        Chemin du fichier CSV à lire.
    delimiter : str, optional
        Délimiteur de colonnes. Doit correspondre à celui utilisé lors
        de l'écriture. Par défaut ``","``.

    Returns
    -------
    List[MetricAccumulated]
        Liste de dictionnaires ``{metric, algo, value}`` reconstituée,
        avec les valeurs converties en ``float`` (ou ``None`` si vide).

    Raises
    ------
    FileNotFoundError
        Si le fichier n'existe pas.
    ValueError
        Si le fichier ne contient pas les colonnes ``Metric`` et ``Algo``.

    Examples
    --------
    >>> data = load_csv("results.csv")
    >>> display_table(data, METRICS_CONFIGS)
    """
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {input_path}")

    accumulated: List[MetricAccumulated] = []

    with input_path.open("r", newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=delimiter)

        if reader.fieldnames is None or "Metric" not in reader.fieldnames:
            raise ValueError(
                f"Le fichier {input_path} ne semble pas être un CSV de métriques valide."
            )

        image_cols = [f for f in reader.fieldnames if f not in {"Metric", "Algo"}]

        for row in reader:
            values: List[Optional[float]] = [
                float(row[col]) if row[col] != "" else None
                for col in image_cols
            ]
            accumulated.append({
                "metric": row["Metric"],
                "algo": row["Algo"],
                "value": values,
            })

    return accumulated
