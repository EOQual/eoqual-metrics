"""
Calcul de métriques No-Reference par tuiles — usage optionnel.

Ce module est **entièrement indépendant** de :class:`~eoqual.metrics.runner.MetricsRunner`
et n'est jamais appelé automatiquement : le calcul par tuiles est une
décision explicite de l'appelant (``compute_score_on_tiles(...)``), jamais
un comportement implicite d'une métrique existante. Reprend l'esprit de
l'ancien moteur de tuilage de ``sharpness_metrics`` (indice de
représentativité par tuile, log résumé plutôt que par-tuile) sur un
principe générique : n'importe quelle métrique No-Reference d'eoqual-metrics
(``sharpness``, ``brisque``, ``niqe``, ``piqe``, ``entropy``, ``fmeasure``...)
peut être tuilée, du moment qu'elle respecte la signature
``f(image, **kwargs) -> float`` ou ``f(image, **kwargs) -> (float, dict)``.

Pourquoi tuiler
---------------
Sur une scène large (typiquement satellite), un score unique moyenne toute
hétérogénéité locale — un flou de bougé localisé, une zone nuageuse, une
plage d'eau sans structure exploitable pour ``aem``/``sasbem`` peuvent
passer inaperçus dans un score global. Le calcul par tuiles produit une
carte de scores, avec un indicateur de fiabilité par tuile.

Exemple
-------
>>> import numpy as np
>>> from eoqual.metrics.nr.sharpness.backends.satellite.aem import aem
>>> image = np.random.rand(1024, 1024).astype(np.float32)
>>> df = compute_score_on_tiles(
...     image, aem, tile_size=256, overlap=0.25,
...     method_kwargs={"return_details": True},
...     metric_name="aem",
... )
>>> df.columns.tolist()
['row', 'col', 'row_end', 'col_end', 'score', 'representativeness', 'insufficient']

Pour une métrique qui ne renvoie qu'un score scalaire (pas de
``return_details``), la colonne ``representativeness`` retombe sur la
variance de la tuile (comme dans ``sharpness_metrics`` à l'origine) et
``insufficient`` reste toujours ``False`` :

>>> from functools import partial
>>> import eoqual.metrics as m
>>> df = compute_score_on_tiles(image, partial(m.sharpness, algo="tenengrad_otsu"))

Limites / évolutions non implémentées ici
------------------------------------------
- **Parallélisation** : chaque tuile est calculée séquentiellement. Un
  ``concurrent.futures.ThreadPoolExecutor`` sur les tuiles serait le levier
  naturel pour les métriques coûteuses (``aem``/``sasbem`` sont déjà
  listées dans ``runner.py::_SLOW_METRICS``) — **volontairement non mis en
  place** : ``aem``/``sasbem`` sont encore à un stade de développement
  (voir leur documentation), paralléliser leur exécution avant qu'elles
  soient stabilisées ajouterait de la complexité (gestion d'erreurs
  concurrentes, ordre non déterministe des logs) pour un gain prématuré.
  À réévaluer une fois ces algorithmes stabilisés — le point d'insertion
  serait la boucle de :func:`compute_score_on_tiles` ci-dessous, en
  remplaçant la boucle ``for`` par une soumission au pool et une collecte
  des résultats dans l'ordre des tuiles (``executor.map`` préserve l'ordre).
- **Intégration à `MetricsRunner`** : non prévue à ce stade. Un paramètre
  ``tile_size=`` optionnel sur le runner est envisageable mais change son
  contrat de résultat (actuellement un score par métrique, pas une carte) —
  à concevoir séparément si le besoin se confirme.
- **Full-Reference** : ce module ne traite que les métriques No-Reference
  (une image). Les métriques FR (GT + P) ont déjà des cartes pixel-à-pixel
  via ``return_map=True`` sur certaines d'entre elles ; un tuilage FR dédié
  n'a pas été demandé et n'est pas couvert ici.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from loguru import logger

MetricFn = Callable[..., Union[float, Tuple[float, dict]]]

# Recouvrement par métrique : comment juger qu'une tuile est
# "représentative" au-delà de la variance générique par défaut. Reprend le
# principe de ``sharpness_metrics._REPRESENTATIVENESS_EXTRACTORS`` — pour
# les méthodes à base d'arêtes, c'est le nombre d'arêtes mesurées qui
# indique la fiabilité, pas la variance du signal.
_REPRESENTATIVENESS_OVERRIDES: Dict[str, Callable[[dict], float]] = {
    "aem": lambda d: float(d.get("n_edges", 0)),
    "sasbem": lambda d: float(d.get("n_edges", 0)) * float(d.get("reliability", 1.0)),
    "mtf": lambda d: float(d.get("n_edges", 0)),
    "blur_kernel": lambda d: 1.0 / (float(d.get("fwhm", 0.0)) + 1e-6),
}


def _tile_bounds(size: int, tile_size: int, overlap: float) -> List[Tuple[int, int]]:
    """
    Positions de départ/fin d'un découpage 1D en tuiles chevauchantes.

    Si ``size <= tile_size``, renvoie une unique tuile ``[0, size)``
    (l'image entière tient dans une seule tuile — pas d'erreur).
    """
    if size <= tile_size:
        return [(0, size)]

    stride = max(1, round(tile_size * (1 - overlap)))
    starts = list(range(0, size - tile_size + 1, stride))
    if starts[-1] != size - tile_size:
        starts.append(size - tile_size)  # dernière tuile alignée sur le bord
    return [(s, s + tile_size) for s in starts]


def compute_score_on_tiles(
    image: npt.NDArray,
    metric_fn: MetricFn,
    tile_size: int = 256,
    overlap: float = 0.25,
    method_kwargs: Optional[dict] = None,
    representativeness: Optional[Callable[[dict], float]] = None,
    metric_name: Optional[str] = None,
) -> pd.DataFrame:
    """
    Calcule une métrique No-Reference sur un découpage en tuiles chevauchantes.

    Parameters
    ----------
    image : npt.NDArray
        Image en niveaux de gris, shape ``(H, W)``.
    metric_fn : Callable
        Fonction de métrique à appliquer à chaque tuile, signature
        ``f(image, **method_kwargs) -> float`` ou
        ``f(image, **method_kwargs) -> (float, dict)``. Peut être la
        fonction d'un backend directement (ex.
        :func:`~eoqual.metrics.nr.sharpness.backends.satellite.aem.aem`,
        avec ``method_kwargs={"return_details": True}`` pour obtenir
        l'indice de fiabilité natif de la métrique), ou toute autre
        fonction respectant cette signature — y compris une métrique
        figée par ``functools.partial`` (ex.
        ``partial(eoqual.metrics.sharpness, algo="tenengrad_otsu")``).
    tile_size : int, optional
        Taille des tuiles carrées, en pixels. Par défaut ``256``. Si
        l'image est plus petite que ``tile_size`` dans une dimension,
        une seule tuile couvrant toute cette dimension est utilisée (pas
        d'erreur, pas de padding).
    overlap : float, optional
        Fraction de recouvrement entre tuiles adjacentes, dans ``[0, 1)``.
        Par défaut ``0.25``.
    method_kwargs : dict, optional
        Arguments additionnels transmis à ``metric_fn`` à chaque appel
        (ex. ``{"return_details": True}``, ou des paramètres assouplissant
        les critères de sélection d'arêtes pour ``aem``/``sasbem`` — voir
        leur documentation).
    representativeness : Callable[[dict], float], optional
        Fonction ``details -> float`` calculant l'indice de
        représentativité d'une tuile à partir de son dictionnaire de
        détail (nécessite ``method_kwargs={"return_details": True}`` pour
        que ``metric_fn`` renvoie effectivement un ``dict``). Si
        ``None`` : recouvrement automatique via ``metric_name``
        (voir :data:`_REPRESENTATIVENESS_OVERRIDES`), sinon variance de
        la tuile (comportement par défaut de ``sharpness_metrics``).
    metric_name : str, optional
        Nom de la métrique (ex. ``"aem"``, ``"sasbem"``) — utilisé pour
        choisir automatiquement une fonction de représentativité adaptée
        (voir ci-dessus) et pour le message de log résumé.

    Returns
    -------
    pd.DataFrame
        Une ligne par tuile : ``row``, ``col`` (coin haut-gauche),
        ``row_end``, ``col_end`` (coin bas-droit), ``score``,
        ``representativeness``, ``insufficient`` (``True`` si le
        dictionnaire de détail de la tuile contient ``"insufficient":
        True`` — sinon toujours ``False``). Les colonnes de détail
        additionnelles (ex. ``n_edges``, ``reliability``, ``mtf50``) sont
        ajoutées si ``metric_fn`` les expose.

    Examples
    --------
    >>> import numpy as np
    >>> from eoqual.metrics.nr.sharpness.backends.satellite.aem import aem
    >>> image = np.random.rand(1024, 1024).astype(np.float32)
    >>> df = compute_score_on_tiles(
    ...     image, aem, tile_size=256,
    ...     method_kwargs={"return_details": True}, metric_name="aem",
    ... )
    >>> df["score"].median()  # doctest: +SKIP

    Notes
    -----
    Séquentiel — voir la docstring du module pour la piste de
    parallélisation (``ThreadPoolExecutor``), documentée mais non mise en
    place tant que ``aem``/``sasbem`` restent en développement.
    """
    if not (0.0 <= overlap < 1.0):
        raise ValueError(f"'overlap' doit être dans [0, 1), reçu : {overlap!r}")

    kwargs = method_kwargs or {}
    rep_fn = representativeness or _REPRESENTATIVENESS_OVERRIDES.get(metric_name or "")

    h, w = image.shape[:2]
    row_bounds = _tile_bounds(h, tile_size, overlap)
    col_bounds = _tile_bounds(w, tile_size, overlap)

    rows: List[dict] = []
    n_insufficient = 0

    for row, row_end in row_bounds:
        for col, col_end in col_bounds:
            tile = image[row:row_end, col:col_end]
            result = metric_fn(tile, **kwargs)

            if isinstance(result, tuple):
                score, details = result
            else:
                score, details = result, {}

            insufficient = bool(details.get("insufficient", False))
            if insufficient:
                n_insufficient += 1

            if rep_fn is not None and details:
                rep = float(rep_fn(details))
            else:
                rep = float(np.var(tile))

            entry = {
                "row": row, "col": col, "row_end": row_end, "col_end": col_end,
                "score": score, "representativeness": rep, "insufficient": insufficient,
            }
            for key, value in details.items():
                if key not in entry and key != "score":
                    entry[key] = value
            rows.append(entry)

    if n_insufficient > 0:
        label = metric_name or getattr(metric_fn, "__name__", "métrique")
        total = len(rows)
        logger.warning(
            f"[{label}] {n_insufficient}/{total} tuile(s) "
            f"({n_insufficient / total:.0%}) insuffisante(s) — "
            f"voir la colonne 'insufficient' pour les filtrer en aval."
        )

    return pd.DataFrame(rows)
