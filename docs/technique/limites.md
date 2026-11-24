# Reduced-Reference, limites transverses et pistes

## 4. Reduced-Reference (RR)

**Aucune métrique implémentée à ce jour** (README, section « Reduced-Reference
IQA » : *None*). Catégorie documentée dans le registre conceptuel mais pas
dans `METRICS_CONFIGS` — à considérer si un besoin se présente (ex.
comparaison à un jeu de statistiques de référence plutôt qu'à l'image
complète, utile quand le GT complet n'est pas disponible/transmissible).

## 5. Limites transverses

### 5.1 Gestion des erreurs du runner

`MetricsRunner.compute()` (`runner.py`) laisse volontairement remonter
toute exception (`try/except` commenté) tant que la librairie n'est pas
jugée stable — choix assumé du mainteneur : une trace complète
(fichier/ligne/cause) est plus utile pour diagnostiquer qu'un `value=None`
silencieux. Conséquence pratique : un batch multi-images s'arrête net à
la première métrique qui échoue (dépendance manquante, image incompatible...).
À réévaluer une fois la robustesse sur de gros lots jugée prioritaire par
rapport au diagnostic fin (voir l'annexe en fin de document).

### 5.2 Licences tierces

Voir **`THIRD_PARTY_LICENSES.md`** pour l'audit complet. Aucun algorithme
obtenu par défaut n'est concerné : les quelques algorithmes 🟡 sous licence
GPL (`metrikz` pour `wsnr`/alt., `ssim_1..4`) sont tous optionnels,
jamais atteints sans les demander explicitement via `algo=`.

### 5.3 Dépendances lourdes / fragiles

`torch`/`torchvision`/`piq`/`IQA_pytorch` : extra optionnel `deep`, version
torch bornée (`<2.3.1`) sur la base des limitations documentées d'
IQA_pytorch (torch ≥ 2.3.1 casse son algo `cw_ssim`, non par défaut).

`piq` (Apache-2.0) est le backend par défaut pour
`dists`/`fsim`/`gmsd`/`vif`/`ssim`/`msssim`/`vsi`/`lpips_vgg` — voir
[§2.3](full-reference.md#23-famille-ssim-similarité-structurelle)/[§2.5](full-reference.md#25-perceptuelle-deep-learning-et-modèles-physiologiques) et `THIRD_PARTY_LICENSES.md`. IQA_pytorch (non maintenu depuis
~2021, MIT) reste disponible en algo legacy partout, et c'est le seul
backend pour `nlpd`. `pyiqa` a été évalué comme alternative plus complète
(couvre aussi `nlpd`/`mad`) et écarté : licence
PolyForm-Noncommercial-1.0.0, non open-source (voir
`THIRD_PARTY_LICENSES.md` [§2.3](full-reference.md#23-famille-ssim-similarité-structurelle)).

### 5.4 Couverture de tests

Trois catégories distinctes, à ne pas confondre :

- **`examples/`** — usage documenté, à jour, destiné à être lu/exécuté par
  un utilisateur : `all_metrics_overview.py` (registre complet),
  `one_metric_cli.py` (CLI une métrique/une image), `tiled_vs_full_image.py`
  (comparaison calcul pleine image vs. par tuiles, voir [§3.4](no-reference.md#34-calcul-par-tuiles-no-reference-opt-in)).
- **`tests/exploratory/`** — scripts d'investigation ponctuelle, pas
  maintenus, pas des exemples à suivre (API tierce appelée directement,
  hors registre eoqual-metrics) : `recover_iqa_pytorch.py`,
  `sewar_overview.py`, `pyiqa_rejected_exploration.py` (documente pourquoi
  `pyiqa` a été écarté, voir `THIRD_PARTY_LICENSES.md` [§2.3](full-reference.md#23-famille-ssim-similarité-structurelle)).
- **`tests/`** — ne garde que `images/` (fixtures) et le vérificateur de
  cohérence `__all__`/`METRICS_CONFIGS` embarqué dans
  `examples/all_metrics_overview.py`. Réservé aux futurs tests `pytest`
  réels — voir `CONTRIBUTING.md` pour contribuer une suite.

**Une vraie suite `pytest` n'existe pas encore**, volontairement : le
registre continue d'évoluer (`aem`/`sasbem` encore en développement), et
une suite assertive écrite maintenant demanderait
d'être retouchée à chaque évolution de ce type — un coût d'entretien
prématuré par rapport à la valeur immédiate. `quality/run_tests.sh` reste
pointé sur `tests/` (commentaire à jour dans le script) : il ne collecte
rien pour l'instant, sans être cassé pour autant — prêt à l'emploi le jour
où une suite y sera ajoutée. Le workflow CI
(`.github/workflows/docs.yml`) construit et publie la documentation
(Sphinx + autoapi) sur GitHub Pages à chaque push sur `main`, mais ne
lance aucun test.

## 6. Pistes

Rien ici ne décrit le comportement actuel de la librairie. Par ordre de valeur/risque décroissant, à discuter et prioriser
avec le mainteneur — rien ci-dessous n'est engagé. Voir `CONTRIBUTING.md`
pour proposer ou implémenter l'un de ces points, ou tout autre bug/métrique :

1. **Remédiation licences des algorithmes optionnels restants** — les 5
   algorithmes 🟡 non par défaut (`metrikz`, `ssim_1..4`, voir
   `THIRD_PARTY_LICENSES.md` [§2.1](full-reference.md#21-erreur-pixel-à-pixel)) pourraient être réécrits en clean-room
   ou retirés — impact nul sur l'usage standard, non engagé.
2. **Alternative permissive pour `nlpd`/`mad`** — `piq` ne les implémente
   pas ; `nlpd` reste fonctionnel via IQA_pytorch (legacy), `mad` reste
   inutilisable quelle que soit la dépendance (`'algo': []`).
3. **Parallélisation du calcul par tuiles** (`ThreadPoolExecutor`,
   `tiling.py`) — différée tant qu'`aem`/`sasbem` ne sont pas stabilisées
   (voir la docstring du module pour le point d'insertion exact).
4. **Intégration du tuilage à `MetricsRunner`** — actuellement une
   fonction autonome jamais appelée implicitement ; changer le contrat de
   résultat du runner (un score, pas une carte) est un chantier séparé.
5. **Suite de tests assertive** par métrique (valeurs de référence
   connues, cas limites — image constante, bruit pur, arêtes absentes) ;
   CI exécutant effectivement `pytest` (voir §5.4).
6. **Réactivation du try/except du runner** (§5.1) une fois la robustesse
   sur de gros lots jugée prioritaire — actuellement report volontaire.
7. **Reduced-Reference** : famille à définir si un besoin se présente
   (§4).
8. **`brisque/numpy_impl.py`** recharge le modèle SVR à chaque appel — un
   cache module-level éviterait de reparser `allmodel` (770 vecteurs de
   support) à chaque image sur de gros lots.
