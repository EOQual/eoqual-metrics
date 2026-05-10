"""
Orchestrateur de calcul de métriques image.

Ce module expose :class:`MetricsRunner`, le point d'entrée pour calculer
un ensemble de métriques sur une ou plusieurs paires d'images, accumuler
les résultats et les transmettre aux modules d'export.

Exemple d'utilisation
---------------------
Calcul sur une seule paire d'images :

>>> runner = MetricsRunner({'psnr': 'numpy', 'ssim': 'skimage'})
>>> results = runner.compute(GT, P)

Calcul sur plusieurs images (accumulation) :

>>> runner = MetricsRunner({})   # toutes les métriques par défaut
>>> for gt, p in image_pairs:
...     runner.compute(gt, p)
...     runner.accumulate()
>>> from eoqual.metrics.export.console import display_table
>>> display_table(runner.results_accumulated, METRICS_CONFIGS)
"""

from __future__ import annotations

import sys
from time import time
from typing import Dict, List, Optional, Union

import numpy as np
import numpy.typing as npt
from loguru import logger
from rich.table import Table
from rich.console import Console

from .config import METRICS_CONFIGS

# Type alias pour un résultat unitaire
MetricResult = Dict[str, Union[str, float, None, npt.NDArray]]
# Type alias pour un résultat accumulé (multi-images)
MetricAccumulated = Dict[str, Union[str, List[Optional[float]]]]

# Métriques connues pour leur temps de calcul long
_SLOW_METRICS: frozenset = frozenset({"cw_ssim", "fsim", "vif", "reco"})


def list_metrics() -> None:
    """Affiche toutes les métriques disponibles et leurs algorithmes."""
    console = Console()
    table = Table(title="Métriques disponibles dans eoqual_metrics", header_style="bold magenta")
    table.add_column("Métrique",     style="bold cyan",   no_wrap=True)
    table.add_column("Mode",         justify="center")
    table.add_column("Algo défaut",  style="green")
    table.add_column("Tous les algos disponibles")
    table.add_column("↓ Mieux",      justify="center")

    for name, cfg in METRICS_CONFIGS.items():
        algos    = ", ".join(a["name"] + (f" ({a['only']})" if a.get("only") else "") for a in cfg["algo"])
        mode     = cfg["metric_mode"]
        default  = cfg["default"]
        lb       = "↓" if cfg["lower_better"] is True else ("↑" if cfg["lower_better"] is False else "?")
        table.add_row(name, mode, default, algos, lb)

    console.print(table)


class MetricsRunner:
    """
    Orchestre le calcul d'un ensemble de métriques image.

    Le runner maintient deux collections internes :

    * :attr:`results` — résultats de la dernière paire calculée
      (valeur scalaire + éventuellement carte pixel-à-pixel).
    * :attr:`results_accumulated` — historique des valeurs scalaires sur
      toutes les paires traitées depuis la création du runner.

    Parameters
    ----------
    bands : str
        Type de données à traiter : 'GRAY' ou 'RGB'. Certaines métriques
        ne sont disponibles que pour l'un ou l'autre. Les algorithmes
        incompatibles avec le mode choisi sont automatiquement exclus du plan
        d'exécution.
    metrics : Dict[str, str]
        Dictionnaire ``{nom_métrique: nom_algo}`` spécifiant les métriques
        à calculer et l'algorithme à utiliser pour chacune.
        Si vide (``{}``), toutes les métriques du registre
        :data:`~eoqual.metrics.config.METRICS_CONFIGS` sont incluses avec
        leur algorithme par défaut.
    exclude : List[str], optional
        Liste de noms de métriques à exclure, uniquement utilisée quand
        ``metrics`` est vide. Par défaut ``[]``.
    default_only : bool, optional
        Si ``True`` (défaut) et ``metrics`` est vide, seul l'algorithme
        par défaut de chaque métrique est sélectionné. Si ``False``, tous
        les algorithmes disponibles sont sélectionnés.

    Raises
    ------
    ValueError
        Si une métrique ou un algorithme spécifié dans ``metrics``
        n'existe pas dans le registre.

    Examples
    --------
    >>> runner = MetricsRunner({'psnr': 'numpy', 'ssim': 'skimage'})
    >>> results = runner.compute(GT, P)
    >>> print(results[0]['value'])

    >>> runner = MetricsRunner({}, exclude=['brisque', 'niqe'])
    >>> for gt, p in pairs:
    ...     runner.compute(gt, p)
    ...     runner.accumulate()
    """

    def __init__(
        self,
        bands: str,
        metrics: Dict[str, str],
        exclude: List[str] = [],
        default_only: bool = True,
    ) -> None:
        
        self.bands = bands
        self.metrics = metrics
        self.exclude = exclude
        self.default_only = default_only

        self.results: List[MetricResult] = []
        self.results_accumulated: List[MetricAccumulated] = []

        self._build_plan()

    # ------------------------------------------------------------------
    # Construction du plan d'exécution
    # ------------------------------------------------------------------

    def _build_plan(self) -> None:
        """
        Construit la liste ordonnée des métriques/algos à calculer.
        """
        if not self.metrics:
            logger.debug("Construction du plan depuis le registre complet...")
            self._build_plan_from_registry()
        else:
            logger.debug("Construction du plan depuis le dictionnaire utilisateur...")
            self._build_plan_from_dict()

    def _build_plan_from_registry(self) -> None:
        """
        Construit le plan depuis le registre complet.
        """
        for metric_name, detail in METRICS_CONFIGS.items():
            if metric_name in self.exclude:
                continue
            for algo in detail["algo"]:
                if self.default_only and algo["name"] != detail["default"]:
                    continue
                if self._is_supported(self.bands, algo):
                    self._register(metric_name, algo)

    def _build_plan_from_dict(self) -> None:
        """
        Construit le plan depuis un dictionnaire utilisateur.
        """
        for metric_name, algo_name in self.metrics.items():
            if metric_name not in METRICS_CONFIGS:
                logger.warning(f"Métrique inconnue ignorée : [{metric_name}]")
                continue
            algo = self._find_algo(metric_name, algo_name)
            if algo is None:
                logger.warning(
                    f"Algorithme [{algo_name}] introuvable pour [{metric_name}], ignoré."
                )
                continue
            if not self._is_supported(self.bands, algo):
                logger.warning(
                    f"[{metric_name}/{algo_name}] non supporté en {self.bands}, cela sera ignoré."
                )
                continue
            self._register(metric_name, algo)

    def _find_algo(
        self, metric_name: str, algo_name: str
    ) -> Optional[Dict]:
        """
        Recherche la définition d'un algo dans le registre.
        """
        for algo in METRICS_CONFIGS[metric_name]["algo"]:
            if algo["name"] == algo_name:
                return algo
        return None

    @staticmethod
    def _is_supported(bands: str, algo: Dict) -> bool:
        """
        Retourne ``True`` si l'algo est supporté par la bands.
        """
        return (bands in algo.values()) or (algo.get("only") is None)

    def _register(self, metric_name: str, algo: Dict) -> None:
        """Enregistre une entrée dans les deux collections internes."""
        entry: MetricResult = {
            "metric": metric_name,
            "algo": algo["name"],
            "value": None,
        }
        if "map" in algo:
            entry["map"] = None
        self.results.append(entry)
        self.results_accumulated.append({
            "metric": metric_name,
            "algo": algo["name"],
            "value": [],
        })

    # ------------------------------------------------------------------
    # Calcul
    # ------------------------------------------------------------------

    def compute(
        self,
        GT: npt.NDArray,
        P: npt.NDArray,
    ) -> List[MetricResult]:
        """
        Calcule toutes les métriques planifiées sur une paire d'images.

        Les résultats sont stockés dans :attr:`results` et retournés.
        Pour accumuler les valeurs scalaires dans :attr:`results_accumulated`,
        appeler :meth:`accumulate` après chaque appel à :meth:`compute`.

        Parameters
        ----------
        GT : npt.NDArray
            Image de référence (ground truth).
        P : npt.NDArray
            Image traitée / dégradée.

        Returns
        -------
        List[MetricResult]
            Liste de dictionnaires ``{metric, algo, value[, map]}``.
            ``value`` est ``None`` si le calcul a échoué ou retourné ``None``.
        """
        import sys
        _pkg = sys.modules[__package__]  # eoqual.metrics already loaded

        tic_total = time()

        for entry in self.results:
            metric_name: str = entry["metric"]   # type: ignore[assignment]
            algo_name: str = entry["algo"]        # type: ignore[assignment]
            mode: str = METRICS_CONFIGS[metric_name]["metric_mode"]

            tic = time()
            logger.debug(f"Calcul : {metric_name} / {algo_name}")

#            try:
            fn = getattr(_pkg, metric_name)
            logger.debug(f"Appel de {fn.__module__}.{fn.__name__}()")
            if mode == "FR":
                result = fn(GT, P, algo=algo_name)
            else:
                result = fn(P, algo=algo_name)

            if isinstance(result, tuple):
                entry["value"], entry["map"] = result  # type: ignore[assignment]
            else:
                entry["value"] = result

#            except Exception as exc:
#                logger.error(
#                    f"Erreur lors du calcul de [{metric_name}/{algo_name}] : {exc}"
#                )
#                entry["value"] = None

            if metric_name in _SLOW_METRICS:
                elapsed = time() - tic
                logger.debug(f"{metric_name} : {elapsed:.2f}s")

        logger.debug(f"Durée totale : {time() - tic_total:.2f}s")
        return self.results

    # ------------------------------------------------------------------
    # Accumulation
    # ------------------------------------------------------------------

    def accumulate(self, precision: int = 4) -> List[MetricAccumulated]:
        """
        Ajoute les valeurs scalaires courantes à l'historique accumulé.

        À appeler après chaque :meth:`compute` dans une boucle multi-images.

        Parameters
        ----------
        precision : int, optional
            Nombre de décimales de l'arrondi. Par défaut ``4``.

        Returns
        -------
        List[MetricAccumulated]
            Historique complet des valeurs accumulées.

        Notes
        -----
        Les valeurs ``None`` (calcul échoué ou métrique non applicable) sont
        conservées telles quelles dans l'historique pour ne pas fausser les
        statistiques aval.
        """
        for entry, accumulated in zip(self.results, self.results_accumulated):
            value = entry["value"]
            if value is not None:
                # Double arrondi pour garantir la troncature exacte
                rounded = float(f"{np.round(float(value), precision):.0{precision}f}")
            else:
                rounded = None
            accumulated["value"].append(rounded)  # type: ignore[union-attr]

        return self.results_accumulated

    # ------------------------------------------------------------------
    # Accès aux données
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """
        Réinitialise les valeurs calculées sans modifier le plan d'exécution.

        Les valeurs dans :attr:`results` sont remises à ``None``.
        L'historique :attr:`results_accumulated` est vidé.
        """
        for entry in self.results:
            entry["value"] = None
            if "map" in entry:
                entry["map"] = None
        for accumulated in self.results_accumulated:
            accumulated["value"] = []  # type: ignore[assignment]

    @property
    def metric_names(self) -> List[str]:
        """Retourne la liste ordonnée des noms de métriques planifiées."""
        return [str(e["metric"]) for e in self.results]

    @property
    def is_empty(self) -> bool:
        """``True`` si aucune métrique n'est planifiée."""
        return len(self.results) == 0
