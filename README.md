# things.encoding

Pipeline d'**encodage cérébral** : prédire l'activité IRMf mesurée pendant le protocole
THINGS de [CNeuroMod](https://www.cneuromod.ca/) à partir des représentations latentes du
modèle vidéo **TRIBE v2** (Meta AI).

Concrètement, on montre à TRIBE v2 exactement les mêmes vidéos que celles vues par les
sujets dans le scanner, on récupère les activations internes du modèle, et on entraîne une
**régression Ridge** par voxel (ou par parcelle) pour prédire le signal BOLD. La qualité de
la prédiction se lit en R² et en corrélation de Pearson.

La question centrale du projet n'est pas seulement « quel R² atteint-on ? », mais
**« combien de ce R² est réel ? »**. Un enregistrement IRMf est fortement autocorrélé dans
le temps : si le découpage train/test coupe au milieu d'un run, le modèle retrouve dans le
test des instants quasi identiques à ceux qu'il a vus en entraînement, et le score monte
sans qu'aucune généralisation n'ait eu lieu. Le projet implémente donc **quatre schémas de
validation croisée** différents et compare ce qu'ils donnent (cf. [Validation croisée](#validation-croisée)).

---

## Sommaire

- [Vue d'ensemble du pipeline](#vue-densemble-du-pipeline)
- [Données](#données)
- [Installation](#installation)
- [Lancer le pipeline](#lancer-le-pipeline)
- [Structure du code](#structure-du-code)
- [Alignement temporel TRIBE ↔ IRMf](#alignement-temporel-tribe--irmf)
- [Validation croisée](#validation-croisée)
- [Sorties et figures](#sorties-et-figures)
- [Continuer le projet](#continuer-le-projet)
- [Faire tourner la Ridge sur GPU](#faire-tourner-la-ridge-sur-gpu)
- [Qualité et style du code](#qualité-et-style-du-code)

---

## Vue d'ensemble du pipeline

```
 ┌─ 1. PRÉPARATION VIDÉO ────────────────────────────────────────────────┐
 │  data/things_mp4_vfr/  (framerate variable)                           │
 │            │  VFRtoCFRConverter.py  —  ffmpeg, fps=64, crf=20         │
 │            ▼                                                          │
 │  data/things_mp4_cfr/  (framerate constant, 834 runs)                 │
 └───────────────────────────────────────────────────────────────────────┘
                             │
 ┌─ 2. EXTRACTION DES LATENTS ───────────────────────────────────────────┐
 │  main.py --subject sub-XX                                             │
 │    Config          → charge facebook/tribev2                          │
 │    TransformerHooks→ forward hooks sur chaque couche attn / ffn       │
 │    model.predict() → activations capturées au vol                     │
 │    HDF5Writer      → écriture incrémentale                            │
 │            ▼                                                          │
 │  output/features/things_encoding/sub-XX.h5   (~14 Go par sujet)       │
 │     ses-XXX / run-N / preds                                           │
 │     ses-XXX / run-N / encoder_layerK_attn    (n_windows, T, 1152)     │
 │     ses-XXX / run-N / encoder_layerK_ffn     (n_windows, T, 1152)     │
 └───────────────────────────────────────────────────────────────────────┘
                             │
 ┌─ 3. ALIGNEMENT TEMPOREL ──────────────────────────────────────────────┐
 │  TribeHDF5Normalization  —  latents à 2 Hz  →  grille IRMf à 1/1.49 Hz│
 │  + BOLD CNeuroMod (data/timeseries/…)                                 │
 │            ▼                                                          │
 │  X (n_TR, 1152)   Y (n_TR, n_parcelles|n_voxels)   groupes (sessions) │
 └───────────────────────────────────────────────────────────────────────┘
                             │
 ┌─ 4. RIDGE + VALIDATION CROISÉE ───────────────────────────────────────┐
 │  RidgeRegression        → un alpha par voxel, 4 schémas de CV         │
 │  VisualisationResultats → cartes cérébrales, histogrammes, accuracy   │
 │            ▼                                                          │
 │  output/analysis/planche_<methode>_<scope>_<sujet>_<couche>.png       │
 └───────────────────────────────────────────────────────────────────────┘
```

---

## Données

Rien de tout cela n'est versionné (`data/` et `output/` sont dans `.gitignore`) : les
volumes vont de quelques Go à 16 Go par fichier.

### Stimuli vidéo

| Dossier | Contenu | Nommage |
|---|---|---|
| `data/things_mp4_vfr/sub-XX/ses-XXX/` | vidéos sources, framerate **variable** | `sub-XX_ses-XXX_task-thingsmemory_run-N.mp4` |
| `data/things_mp4_cfr/` (plat) | vidéos converties, framerate **constant** | `sub-XX_ses-XXX_task-thingsmemory_run-N_desc-CFR.mp4` |

834 runs CFR au total, pour les sujets `sub-01`, `sub-02`, `sub-03`, `sub-06`.
Chaque run VFR est accompagné de son `*_events.tsv` (onsets, offsets, image présentée).

### Timeseries IRMf

Elles proviennent du dataset DataLad [`courtois-neuromod/things.timeseries`](https://github.com/courtois-neuromod/things.timeseries)
(déjà détrendées, lissées, masquées et vectorisées). Deux précisions sont supportées :

| Précision | Fichier | Dimension de Y |
|---|---|---|
| **Parcelles** (défaut) | `data/timeseries/cneuromod2026/sub-XX/sub-XX_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5` | 1134 parcelles |
| **Voxels** | `data/timeseries/voxel_native/sub-XX/sub-XX_task-things_space-T1w_desc-voxelwise_timeseries.h5` | ~10⁵ voxels |

Chaque dossier contient aussi l'image d'atlas (`_dseg.nii.gz`) ou le masque de matière
grise (`_label-GMfromFS_desc-indivFunc_mask.nii.gz`) nécessaires aux cartes cérébrales.

### Masques et annotations — `data/brain_map_subj/`

- `sub-XX_space-T1w_desc-ROImasks_voxelAnnotations.h5` — ROIs voxelwise, regroupées en
  `retinotopy_ROIs` (V1, V2, V3, hV4, LO1…), `fLoc_ROIs` (FFA, OFA, PPA, EBA…) et `yeo_ROIs`.
- `tpl-MNI152NLin2009cAsym_atlas-Schaefer2018TianS3NettekovenAsym_…_parcelAnnotations.tsv` —
  annotations des 1134 parcelles, utilisées pour filtrer par réseau (`Vis`, `DorsAttn`).
  L'ordre des lignes suit l'ordre des colonnes du fichier de timeseries.
- `sub-XX_task-things_stat-noiseCeilings_featureAnnotations.h5` — plafonds de bruit.

### Constantes clés

| Constante | Valeur | Sens |
|---|---|---|
| `T_TRIBE_S` | 0.5 s | période d'échantillonnage des latents TRIBE (2 Hz) |
| `TR_IRMF_S` | 1.49 s | TR de l'acquisition IRMf |
| dimension cachée | 1152 | taille d'un embedding TRIBE |
| `NB_SESSIONS_TOTAL` | 36 | sessions du protocole THINGS |

---

## Installation

### Prérequis

- **Python 3.12** (cf. `.python-version` ; la borne haute est fixée dans `pyproject.toml`)
- [**uv**](https://docs.astral.sh/uv/getting-started/installation/)
- **ffmpeg** et **ffprobe** accessibles dans le `PATH` (conversion vidéo, lecture des durées)
- Un compte HuggingFace avec accès au modèle *gated* [`facebook/tribev2`](https://huggingface.co/facebook/tribev2)
- Un **GPU CUDA** pour l'extraction des latents. L'analyse Ridge tourne sur CPU par défaut,
  et peut être basculée sur GPU — cf. [Faire tourner la Ridge sur GPU](#faire-tourner-la-ridge-sur-gpu)

### Mise en place

```bash
git clone <url-du-repo>
cd things.encoding
uv sync                      # crée .venv/ et installe tout, y compris TRIBE v2 depuis GitHub
```

Créer ensuite un fichier `.env` à la racine :

```bash
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

Le token se récupère sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).
TRIBE v2 embarque **LLaMA 3.2**, lui aussi *gated* : l'accès doit être demandé sur la page
HuggingFace de Meta **avant** le premier téléchargement. Les poids atterrissent dans
`./cache/` (quelques Go) ; les lancements suivants réutilisent ce cache. En l'absence de
réseau, `HF_HUB_OFFLINE=1` fait sauter le `login()` et force l'usage du cache.

---

## Lancer le pipeline

> **Important** — `src/` est un dossier plat, sans `__init__.py`, et les imports internes
> sont nus (`from Config import Config`). Les scripts se lancent donc **depuis `src/`**,
> ou avec `src/` ajouté à `sys.path`.

```bash
cd src
```

| Ordre | Commande | Effet |
|---|---|---|
| 1 | `uv run python VFRtoCFRConverter.py` | convertit `data/things_mp4_vfr/**` en framerate constant (64 fps, CRF 20) |
| 2 | `uv run python main.py --subject sub-01` | extrait les latents TRIBE d'un sujet → `output/features/things_encoding/sub-01.h5` |
| 3 | `uv run python RidgeRegression.py` | **analyse principale** : Ridge + validation croisée + figures |

Points à connaître :

- **Étape 1** : le convertisseur écrit dans `data/things_mp4_cfr/<sujet>/<session>/<nom>_cfr.mp4`,
  alors que l'étape 2 cherche des fichiers `*_desc-CFR.mp4` (à plat dans `data/things_mp4_cfr/`).
  Si les vidéos CFR sont produites localement plutôt que récupérées telles quelles, il faut
  donc les renommer/aplatir pour respecter cette convention.
- **Étape 2** est reprenable : un run dont le groupe `session/run/preds` existe déjà dans le
  HDF5 est ignoré, on peut donc relancer après une interruption. Elle demande un GPU
  (`Config.charger_modele` ne bascule sur CPU que si `plateforme == "Mac"`, or `main.py`
  passe `plateforme=None`).
- **Étape 3 n'a pas d'arguments en ligne de commande** : tout se règle dans le bloc
  `if __name__ == "__main__":` en bas de [`RidgeRegression.py`](src/RidgeRegression.py) —
  voir [Continuer le projet](#continuer-le-projet).

### Points d'entrée annexes

| Commande | Effet |
|---|---|
| `uv run python main_embeddings_friends.py --season 1 --episode 1 --partieEpisode a` | même extraction, sur les épisodes de *Friends* (une demi-épisode par appel, pensé pour un job array SLURM) |
| `uv run python CreateOneImageVideo.py` | expérience contrôle : boucle chaque image THINGS distincte en une vidéo de 100 s |
| `uv run python TribeHDF5Normalization.py` | démonstration de l'alignement sur un seul run |
| `uv run python ../test/test_litcoder_folding.py` | compare les découpages de validation croisée sur la structure réelle des sessions/runs |
| `uv run python ../test/test_equivalence_scopes.py` | vérifie qu'un scope se déduit d'une CV cerveau entier sans la relancer (cf. [Un seul passage pour tous les scopes](#un-seul-passage-pour-tous-les-scopes)) |

---

## Structure du code

Tout tient dans [`src/`](src/), en trois familles.

### Extraction des représentations

| Fichier | Rôle |
|---|---|
| [`Config.py`](src/Config.py) | charge `.env`, s'authentifie auprès de HuggingFace, instancie `TribeModel` (CPU si `plateforme == "Mac"`, CUDA sinon) |
| [`TransformerHooks.py`](src/TransformerHooks.py) | attache des *forward hooks* sur les couches de `FmriEncoderModel` et collecte leurs sorties |
| [`HDF5Writer.py`](src/HDF5Writer.py) | écrit `{sujet}.h5` en mode append, en recréant les datasets déjà présents |
| [`main.py`](src/main.py) | orchestre l'extraction sur toutes les vidéos d'un sujet |
| [`main_embeddings_friends.py`](src/main_embeddings_friends.py) | même chose sur *Friends*, avec chronométrage des étapes |

**Convention de nommage des couches.** `TransformerHooks` parcourt `encoder.layers`, où les
blocs alternent attention et feed-forward. L'indice de couche transformer est donc `i // 2`
et le type dépend de la parité :

```python
layer_type = 'attn' if i % 2 == 0 else 'ffn'
name = f'encoder.layer{i // 2}.{layer_type}'
```

Les points deviennent des underscores dans le HDF5 : `encoder_layer7_ffn` désigne la sortie
du bloc feed-forward de la 8ᵉ couche transformer. C'est la valeur utilisée par défaut dans
l'analyse.

### Préparation vidéo

| Fichier | Rôle |
|---|---|
| [`VFRtoCFRConverter.py`](src/VFRtoCFRConverter.py) | conversion VFR → CFR via ffmpeg (`fps`, `crf` réglables ; CRF contraint à 18-23) |
| [`VideoSegmenteur.py`](src/VideoSegmenteur.py) | découpe un run en clips par image, et boucle un clip jusqu'à une durée cible |
| [`CreateOneImageVideo.py`](src/CreateOneImageVideo.py) | construit les vidéos contrôle de 100 s à partir d'une image fixe |

### Analyse

| Fichier | Classe / fonctions | Rôle |
|---|---|---|
| [`TribeHDF5Normalization.py`](src/TribeHDF5Normalization.py) | `TribeHDF5Normalization` | aligne temporellement un run TRIBE et un run BOLD → `(X, Y)` |
| [`GroupShuffleSplitSession.py`](src/GroupShuffleSplitSession.py) | `GroupShuffleSplitSession` | tirage de **sessions** entières en test |
| [`GroupShuffleSplitRun.py`](src/GroupShuffleSplitRun.py) | `GroupShuffleSplitRun` | tirage de **runs** entiers en test, avec buffer de runs voisins |
| [`litcoder_folding.py`](src/litcoder_folding.py) | `create_folds`, `create_chunked_folds_trimmed`… | portage du *folding* de [`litcoder_core`](https://github.com/GT-LIT-Lab/litcoder_core), plus une option `runs=` propre au projet |
| [`RidgeRegression.py`](src/RidgeRegression.py) | `RidgeRegression`, `CheminsProjet`, `ResultatsCV` | chargement, masquage, entraînement Ridge et validation croisée |
| [`VisualisationResultats.py`](src/VisualisationResultats.py) | `VisualisationResultats` | toutes les figures — **ne calcule rien** |

La séparation entre les deux dernières classes est volontaire et **à sens unique** :
`RidgeRegression` importe `VisualisationResultats`, jamais l'inverse. Les chemins lui sont
donnés à la construction sous forme d'un `CheminsProjet` déjà résolu, et les résultats sous
forme d'un `ResultatsCV` dont elle lit les champs par attribut, sans import.

---

## Alignement temporel TRIBE ↔ IRMf

C'est l'étape la moins devinable du projet. Les latents TRIBE sortent à **2 Hz**, le BOLD est
échantillonné à **1 / 1.49 Hz**, et le signal hémodynamique est décalé de plusieurs secondes
par rapport au stimulus. `TribeHDF5Normalization.executer_pipeline` réconcilie tout ça :

1. **Chargement** de `Y` (BOLD de la session/run) et de `X` (latents de la couche demandée,
   aplatis en `(n_instants, 1152)`).
2. **Coup de ciseaux** — TRIBE produit des fenêtres qui débordent de la vidéo réelle. La
   durée vraie est lue avec `ffprobe`, et seuls les `durée / 0.5` premiers instants sont
   conservés.
3. **Délai hémodynamique**, selon le drapeau `flag_delai_bold_brute` :
   - `True` — **aucune convolution** : on décale simplement l'axe temporel IRMf de −5 s,
     approximation grossière mais robuste du retard BOLD ;
   - `False` — convolution de `X` par la HRF SPM (`nilearn.glm.first_level.spm_hrf`,
     `t_r = 0.5`), avec axes temporels centrés.
   Le drapeau `centrage_donne_temps` ajoute un demi-échantillon de décalage.
4. **Masque de validité** — on ne garde que les TR dont l'instant cible tombe à l'intérieur
   de la plage couverte par les latents.
5. **Rééchantillonnage** sur la grille des TR avec `interp1d(..., kind='previous')` : chaque
   embedding est **répété jusqu'à l'apparition du suivant** plutôt qu'interpolé linéairement
   — c'est la pratique recommandée dans la littérature, un embedding n'ayant pas de sens
   « entre deux valeurs ».
6. **Conversion** de `X` et `Y` en `float32`.

Côté `RidgeRegression.create_X_Y_total`, les runs sont ensuite concaténés, avec :
- l'exclusion manuelle de `sub-06 / ses-08 / run-6` (mauvais alignement spatial après prétraitement) ;
- le saut des runs dont la vidéo ou le dataset BOLD est absent ;
- si `randomize_flag=True`, une **permutation sans point fixe** de l'ordre des runs de `Y`
  (graine 42) qui sert de baseline : le R² obtenu ainsi est le niveau du hasard.

---

## Validation croisée

C'est le cœur méthodologique du projet. Les quatre méthodes partagent la même mécanique
interne — grille `np.logspace(-1, 10, 20)`, **un alpha par voxel**, standardisation de `X` et
`Y` réapprise strictement à l'intérieur du train — et ne diffèrent que par le **découpage
externe**.

### 1. `nested_cross_validation_full_manuel`

Validation croisée imbriquée entièrement explicite : boucle externe (par session ou par run)
et boucle interne `LeaveOneGroupOut` par session, avec un fit Ridge pour chaque alpha × chaque
fold interne. La sélection d'alpha utilise un **R² poolé** (résidus accumulés sur tous les
folds internes, puis un seul calcul de R²), et non une moyenne de R² par fold — c'est ce que
fait réellement `RidgeCV(cv=None)`, et l'objectif est d'en reproduire le mécanisme exactement.
Coûteuse (triple boucle, aucun raccourci algébrique), c'est la référence de vérification.

### 2. `nested_cross_validation_ridgecv_loo`

Jumeau *sklearn-natif* de la précédente : même découpage externe, mais la boucle interne est
remplacée par un `RidgeCV` (cf. `RidgeRegression._ajuster_ridgecv`) :

```python
scaler_Y = StandardScaler()
modele = make_pipeline(
    StandardScaler(),
    RidgeCV(alphas=grille_alphas, alpha_per_target=True, cv=None, scoring="r2"),
)
modele.fit(X_train, scaler_Y.fit_transform(Y_train))
Y_pred = scaler_Y.inverse_transform(modele.predict(X_test))
```

La standardisation de `Y` est explicite plutôt que déléguée à un
`TransformedTargetRegressor` : ce dernier reconvertit `y` en numpy et perd le périphérique,
ce qui rendait le calcul sur GPU impossible. Les opérations et les résultats sont identiques.

`cv=None` déclenche le LOO analytique (validation croisée généralisée), incomparablement plus
rapide. **Nuance importante** : ce LOO se fait *par point temporel*, là où la version manuelle
faisait un LOGO *par session*. Les deux méthodes sont donc comparables en mécanique et en
temps de calcul, mais pas strictement en interprétation scientifique.

C'est la méthode utilisée par défaut, déclinée en deux variantes :

| Variante | Découpage externe | Ce qu'elle mesure |
|---|---|---|
| `niveau_split="session"` | sessions entières en test (`GroupShuffleSplitSession`) | généralisation à un **autre jour de scan** |
| `niveau_split="run"` | runs entiers en test, plus un buffer de `n_buffer` runs voisins (`GroupShuffleSplitRun`) | généralisation à un **autre run** |

**Session ou run ?** Aucun des deux n'est « plus propre » dans l'absolu :

- Le split **par session** n'a pas besoin de buffer : entre deux sessions il y a plusieurs
  jours d'écart, donc ni autocorrélation BOLD ni dérive de scanner commune. Train et test
  partitionnent exactement les échantillons.
- Le split **par run** teste plus finement, mais les *autres* runs de la session testée
  (même jour, même repositionnement de tête, même dérive lente) restent en entraînement. Le
  buffer écarte les runs immédiatement voisins, au prix d'un train plus petit.

Conséquence pratique, écrite noir sur blanc dans la docstring de `GroupShuffleSplitRun` :
**un R² plus élevé au niveau run qu'au niveau session ne signifie pas « meilleur modèle »**.
Pour trancher, on peut activer la variante `n_buffer=0`, qui rend au train sa taille du niveau
session : l'écart qui subsiste alors ne vient plus du volume de données.

### 3. `nested_cross_validation_chunked_trimmed_ridgecv_loo`

Même boucle interne, mais découpage externe issu de `create_chunked_folds_trimmed`
([`litcoder_folding.py`](src/litcoder_folding.py)) : une **vraie partition**, chaque chunk
étant testé exactement une fois. Avec `chunk_length=None`, les chunks sont les **runs réels** —
seule façon de garantir qu'aucun run ne se retrouve à cheval sur train et test. Le paramètre
`trim_size` rogne les bords des chunks de test pour casser l'autocorrélation locale.

### 4. `nested_cross_validation_one_cycle`

Le protocole officiel **CNeuroMod-THINGS**, entièrement figé (aucune graine, aucun tirage) :

```
Test           : sessions 14, 15, 16          (identique pour tous les folds)
Buffer du test : sessions 13, 17              (jamais en train ni en validation)
Validation     : 8 blocs de 3 sessions, chacun avec son propre buffer
Train          : tout le reste, fold par fold
```

Chaque fold sélectionne l'alpha sur son unique split Train → Validation, puis réentraîne sur
Train + Validation et évalue **une seule fois** sur le test figé. Les folds impossibles
(validation ou test absents des données du sujet) sont exclus des agrégations avant le calcul
de la moyenne et de la variance, pour ne pas les tirer vers zéro.

C'est la **seule méthode qui produit aussi un score de corrélation de Pearson**, calculé par
voxel avec `scipy.stats.pearsonr(..., axis=0)`. Le Pearson ignore le biais et l'échelle de la
prédiction, que le R² pénalise : il est donc mécaniquement plus élevé, et se lit comme « la
forme temporelle prédite est-elle la bonne ? », indépendamment de l'amplitude. Il apparaît en
second panneau de la figure *accuracy*, et volontairement **pas** en carte cérébrale, où il
dessinerait la même topographie que le R².

### Un seul passage pour tous les scopes

Un `masque_roi` ne filtre que des **colonnes** de `Y`, jamais des lignes. Or une Ridge
multi-cible régresse chaque voxel indépendamment des autres : `StandardScaler` normalise
colonne par colonne, `alpha_per_target=True` choisit un alpha par cible,
`r2_score(multioutput='raw_values')` score par colonne — et le découpage train/test ne
dépend que des sessions/runs, donc pas du masque.

Il s'ensuit que **le R² d'une parcelle est le même qu'on l'ait calculé sur les 1134
parcelles ou sur 200**. Mesuré sur les deux chemins de modèle du projet : écart maximal de
`2.2e-16` (la précision machine) et alphas strictement identiques.

Le pipeline ne lance donc la validation croisée **qu'une fois**, sur `scope_cv`, et déduit
chaque scope par sélection de colonnes (`ResultatsCV.restreindre`). Ce n'est pas qu'une
économie de régression : `_selection_X_Y` rappelant `create_X_Y_total()` à chaque appel,
on passe de trois relectures du fichier TRIBE de 14 Go à une seule — c'est le poste
dominant du temps d'exécution.

C'est une hypothèse sur le comportement de scikit-learn, pas une garantie d'API : si une
version future couplait les cibles entre elles, les figures deviendraient fausses en
silence. D'où [`test/test_equivalence_scopes.py`](test/test_equivalence_scopes.py), qui
la revérifie.

---

## Sorties et figures

Chaque méthode produit **un seul fichier** (300 DPI) dans `output/analysis/` :
`planche_<methode>_<scope>_<sujet>_<couche>.png`. Toutes les figures ci-dessous y sont
assemblées, et les PNG individuels sont supprimés après coup.

### Le contenu de la planche

`VisualisationResultats.generer_toutes_les_figures` produit la série complète de figures
pour le scope désigné par `scope_detaille` :

1. **Carte cérébrale du R² moyen** — `plot_stat_map` en mosaïque, via un `NiftiLabelsMasker`
   (parcelles) ou un `NiftiMasker` sur fond T1w (voxels).
2. **Carte cérébrale des alphas** — en échelle log, la moyenne inter-folds étant **géométrique**
   (`10 ** mean(log10(alphas))`), les alphas s'étalant sur plusieurs décades.
3. **Histogrammes des alphas** — un par fold externe, puis la distribution des alphas moyens ;
   plus, pour `full_manuel` seule, la distribution des alphas des folds internes.
4. **Accuracy** — barres `mean` / `median` / `top-10% mean` / `max`, chaque barre portant
   l'écart-type **inter-folds** (une valeur par fold, jamais une moyenne unique). Pour
   `one_cycle`, la figure comporte deux panneaux : R² à gauche, Pearson r à droite, sur des
   échelles indépendantes.
5. **Distribution du R²** et **R² au-dessus d'un seuil**.
6. **Score par ROI** — en précision voxel et analyse cerveau entier uniquement. Voir
   [Lire la figure par ROI](#lire-la-figure-par-roi).
7. **Comparaison des scopes** — voir [Lire la comparaison des scopes](#lire-la-comparaison-des-scopes).

Les figures carrées sont rangées sur 3 colonnes ; les deux dernières, plus larges, occupent
chacune une rangée entière — tassées dans une case carrée, leurs étiquettes seraient
illisibles.

Les résultats numériques sont aussi renvoyés sous forme de dictionnaire
(`r2_moyen`, `r2_tous_les_tests`, `alphas_moyens`, `pearson_moyen`) pour comparer les méthodes
entre elles.

### Lire la figure par ROI

`plot_ROImask_histogram` ventile le score moyen sur les 25 ROIs du fichier `ROImask`. Les
ROIs sont **groupées par famille** (blocs de couleur contigus : `early`, `lateral`, `ventral`,
`scene`, `body`, `face`, puis les réseaux Yeo entiers), les familles classées par R² moyen
décroissant et les ROIs décroissantes à l'intérieur de chaque bloc. Chaque barre porte sa
valeur, l'effectif est dans l'étiquette (`V1 (n=1 726)`), et un bloc de texte sous la figure
explicite chaque abréviation.

Pour `one_cycle`, la figure a deux panneaux, **R² à gauche et Pearson r à droite, dans le même
ordre de ROIs**. C'est le point de la figure : les lignes se lisent de gauche à droite, ce qui
fait ressortir les aires où le Pearson est élevé alors que le R² reste faible — celles dont la
forme temporelle est bien prédite, mais pas l'amplitude.

Les réseaux Yeo (`visual`, `defaultMode`…) sont toujours renvoyés en bas du classement, quel
que soit leur score : ce sont des réseaux entiers, pas des aires du système visuel, et les
mêler au tri reviendrait à comparer des objets de nature différente.

### Lire la comparaison des scopes

`plot_comparaison_scopes` met les scopes côte à côte : les mêmes quatre métriques que
l'*accuracy* (`mean`, `median`, `top-10% mean`, `max`), mais une couleur par scope au lieu
d'une figure par scope. Les moustaches restent l'écart-type inter-folds, et l'effectif de
chaque scope est dans la légende. Comme pour l'*accuracy*, un second panneau Pearson apparaît
quand la méthode en produit. Un bloc de texte sous les panneaux **détaille le contenu de
chaque scope** — les 19 aires que recouvre `ROIs` en précision voxel, les réseaux d'atlas en
précision parcelles.

Deux points de lecture :

- **`max` est la seule barre qui ne se compare pas à effectif égal.** Un grand scope contient
  les parcelles d'un petit, donc son maximum lui est mécaniquement supérieur ou égal. Les
  trois autres métriques sont des moyennes, insensibles à l'effectif.
- Les chiffres proviennent d'**un seul passage de validation croisée**, pas d'un par scope
  (cf. [Un seul passage pour tous les scopes](#un-seul-passage-pour-tous-les-scopes)).

---

## Continuer le projet

### Le panneau de commande

Tout se pilote depuis le bloc `if __name__ == "__main__":` de
[`RidgeRegression.py`](src/RidgeRegression.py) :

```python
plateforme            = "Rorqual"          # ou autre valeur → chemins locaux
liste_sujets          = ["sub-03"]
LAYER                 = "encoder_layer7_ffn"
flag_delai_bold_brute = True               # décalage −5 s ; False → convolution HRF
centrage_donne_temps  = False
flag_precision_voxel  = False              # False → 1134 parcelles ; True → voxels
flag_gpu              = False              # True → régressions sur GPU, cf. plus bas
alphas                = np.logspace(-1, 10, 20)
```

`plateforme` ne change qu'une chose : les racines de chemins. Sur `"Rorqual"` elles sont
codées en dur vers `/home/aclaud/links/scratch/…` et le dossier des latents s'appelle `hdf5` ;
partout ailleurs, la racine est celle du dépôt et le dossier s'appelle `features`.
**C'est le premier endroit à adapter pour installer le projet sur une autre machine.**

Viennent ensuite deux dictionnaires :

- **`scopes`** — les zones cérébrales comparées. En parcelles : tout le cerveau, ou les
  réseaux `Vis` / `Vis + DorsAttn` (via `_charger_masque_parcelles`). En voxels : cerveau
  entier, ou l'union des ROIs rétinotopiques et catégorielles (via `_charger_masque_roi`).
- **`methodes`** — les schémas de validation croisée à lancer, sous forme de lambdas. Plusieurs
  variantes sont présentes en commentaire, prêtes à être réactivées.

Et deux réglages qui décident de ce qui est calculé et de ce qui est détaillé :

```python
scope_cv        = None                 # sur quoi la CV tourne réellement
scope_detaille  = "toutes_parcelles"   # quel scope reçoit la planche complète
```

- **`scope_cv`** — `None` (défaut) fait tourner la CV sur tout le cerveau, ce qui rend les
  trois scopes dérivables. Le restreindre à un masque accélère le calcul mais limite la
  comparaison : les scopes qui n'y sont pas inclus sont sautés avec un message, et « tout
  le cerveau » cesse d'être disponible — l'afficher reviendrait à étiqueter le sous-espace
  de la CV comme s'il était complet.
- **`scope_detaille`** — le seul scope à recevoir cartes cérébrales et histogrammes d'alphas.
  Les autres n'apparaissent que dans la figure de comparaison. Il doit être inclus dans
  `scope_cv`, sinon le lancement s'arrête avec une erreur explicite.

Chaque méthode produit donc **une planche détaillée et une figure de comparaison**, quel que
soit le nombre de scopes.

### Faire tourner la Ridge sur GPU

`flag_gpu = True` bascule les régressions sur GPU via le [support de l'API Array de
scikit-learn](https://scikit-learn.org/stable/modules/array_api.html). **Les résultats sont
les mêmes** — alphas identiques, R² et Pearson à la précision `float32` près ; seul le temps
de calcul change.

C'est pensé pour la **précision voxel**. En parcelles, `Y` pèse 0,2 Go et le calcul passe
déjà bien sur CPU ; en voxels il pèse ~17 Go et chaque fold enchaîne une SVD sur
`X (40 470, 1152)` puis un balayage de 20 alphas sur 104 007 cibles.

Le flag est sans danger :

- **aucun GPU disponible** → message explicite, l'analyse continue sur CPU ;
- **mémoire GPU insuffisante** → l'empreinte estimée est affichée, puis repli sur CPU. Mieux
  vaut ça qu'un `CUDA out of memory` au milieu d'un job de plusieurs heures ;
- **sur MPS** (Apple Silicon) `aten::_linalg_eigh` n'est pas implémenté et retombe sur CPU :
  le gain y est moindre que sur CUDA.

Deux variables d'environnement sont nécessaires, et doivent être posées **avant** l'import de
scipy et de torch. `RidgeRegression.py` les pose lui-même en tête de module, ce qui suffit
quand il est le point d'entrée ; dans un script SLURM, mieux vaut les exporter explicitement :

```bash
export SCIPY_ARRAY_API=1              # sans elle, sklearn refuse d'activer l'API Array
export PYTORCH_ENABLE_MPS_FALLBACK=1  # utile sur Mac uniquement, sans effet sur CUDA
```

Aucune dépendance supplémentaire : `torch` est déjà requis par l'extraction, et scikit-learn
1.9 n'a pas besoin de `array-api-compat`.

### Qualité et style du code

Le projet est formaté et vérifié par [ruff](https://docs.astral.sh/ruff/), piloté par
[pre-commit](https://pre-commit.com). La configuration vit dans `[tool.ruff]` de
[`pyproject.toml`](pyproject.toml) et dans [`.pre-commit-config.yaml`](.pre-commit-config.yaml).

```bash
uv run pre-commit install          # une fois par clone : le contrôle tourne à chaque commit
uv run pre-commit run --all-files  # passage manuel sur tout le dépôt
uv run ruff format src test        # formatage seul
uv run ruff check src test --fix   # lint + corrections automatiques
```

Conventions retenues : 88 colonnes, guillemets doubles, indentation à 4 espaces, imports triés
(`I`), et les familles de règles `E`/`F`/`B`/`UP`. `E501` est ignoré — le formateur possède
déjà la longueur de ligne, et la règle ne se déclencherait plus que sur ce qu'il ne peut pas
couper. Deux fichiers ignorent `E402`, avec la raison en commentaire dans `pyproject.toml`.

### Points d'extension

| Objectif | Où intervenir |
|---|---|
| Tester une autre couche TRIBE | changer `LAYER` — toutes les couches sont déjà dans le HDF5, aucune ré-extraction nécessaire |
| Passer en voxelwise | `flag_precision_voxel = True` (fichiers ~16 Go, prévoir la mémoire) ; envisager `flag_gpu = True` |
| Ajouter un schéma de validation croisée | nouvelle méthode `nested_cross_validation_*` renvoyant un `ResultatsCV` ; les champs optionnels laissés à `None` font simplement sauter les figures correspondantes |
| Ajouter une figure | nouvelle méthode de `VisualisationResultats`, appelée depuis `generer_toutes_les_figures` et ajoutée à `liste_chemins_figures` pour entrer dans la planche |
| Comparer d'autres zones cérébrales | ajouter une entrée à `scopes` — aucune CV supplémentaire, le scope est dérivé du passage existant tant qu'il est inclus dans `scope_cv` |
| Ajouter une métrique | la propager via un champ de `ResultatsCV` — le contrat entre les deux classes se lit par attribut, sans import croisé |
| Changer le découpage temporel | `TribeHDF5Normalization` pour l'alignement, les classes `GroupShuffleSplit*` ou `litcoder_folding` pour les folds |

### Piste ouverte : Friends

L'extraction sur les épisodes de *Friends* est fonctionnelle et une saison complète est déjà
encodée dans `output/features/friends/`, avec les timeseries correspondantes dans
`data/friends.timeseries/`. **Aucun chemin de code côté Ridge ne les consomme encore** :
brancher `RidgeRegression` sur ce second dataset est le prolongement naturel du projet.

### Dépendances principales

| Package | Rôle |
|---|---|
| `tribev2` | modèle TRIBE v2 (installé depuis GitHub) |
| `torch`, `torchvision`, `torchcodec` | backend deep learning et décodage vidéo |
| `scikit-learn` | Ridge, RidgeCV, scalers, splitters |
| `scipy` | HRF, interpolation, corrélation de Pearson vectorisée (**≥ 1.13** pour `pearsonr(axis=…)`) |
| `nilearn` | maskers, cartes cérébrales, HRF SPM |
| `h5py` | lecture/écriture des timeseries et des latents |
| `matplotlib`, `seaborn` | figures |
| `ffmpeg-python`, `moviepy`, `opencv-python` | manipulation vidéo |

`exca` est épinglé à `>=0.5.20,<0.5.24` pour rester compatible avec `neuralset`, une
dépendance interne de TRIBE v2. La liste faisant foi est `pyproject.toml`, verrouillée par
`uv.lock`.
