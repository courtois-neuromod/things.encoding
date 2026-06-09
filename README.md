# things.encoding

Pipeline d'extraction et de sauvegarde au format HDF5 des représentations latentes multi-couches du modèle TRIBE v2, à partir de séquences vidéo du dataset THINGS.

## Prérequis

- Python 3.11 ou 3.12
- [uv](https://docs.astral.sh/uv/getting-started/installation/) — gestionnaire de paquets et d'environnements virtuels
- Git
- Un compte [HuggingFace](https://huggingface.co) avec accès au modèle [facebook/tribev2](https://huggingface.co/facebook/tribev2)

## Installation

### 1. Cloner le repo

```bash
git clone https://github.com/cneuromod/things.encoding.git
cd things.encoding
```

### 2. Installer les dépendances

```bash
uv sync
```

Cette commande crée automatiquement le virtualenv dans `.venv/` et installe toutes les dépendances, y compris TRIBE v2 depuis GitHub.

### 3. Configurer le token HuggingFace

Créer un fichier `.env` à la racine du projet :

```bash
HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

Le token est disponible sur [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

### 4. Vérifier l'installation

```bash
uv run python test_single_stimulus.py
```

Le modèle TRIBE v2 se télécharge dans `./cache/` au premier lancement (~quelques Go). Les lancements suivants utilisent le cache local.

## Structure du projet

```
things.encoding/
├── .env                  # variables d'environnement (non versionné)
├── .python-version       # version Python cible (3.11)
├── .venv/                # virtualenv géré par uv (non versionné)
├── cache/                # poids TRIBE v2 téléchargés depuis HuggingFace (non versionné)
├── data/                 # vidéos THINGS en entrée (non versionné)
├── outputs/              # fichiers HDF5 générés (non versionné)
├── pyproject.toml        # dépendances et configuration du projet
├── uv.lock               # lockfile uv (versionné)
├── hello_tribe.py        # script de vérification de l'installation
└── src/
    └── things_encoding/  # code source du pipeline
```

## Commandes utiles

```bash
# Installer / mettre à jour l'environnement
uv sync

# Ajouter une dépendance
uv add <package>

# Lancer un script
uv run python <script.py>

# Lancer Jupyter
uv run jupyter notebook
```

## Dépendances principales

| Package | Rôle |
|---|---|
| `tribev2` | Modèle TRIBE v2 (Facebook Research) |
| `torch` | Backend deep learning |
| `h5py` | Sauvegarde des représentations latentes en HDF5 |
| `opencv-python` | Extraction de frames depuis les vidéos |
| `transformers` | Accès aux encodeurs HuggingFace |
| `huggingface-hub` | Téléchargement des poids du modèle |

## Notes

- Les vidéos THINGS ont un framerate variable.
- `exca` est épinglé à `>=0.5.20,<0.5.24` pour assurer la compatibilité avec `neuralset` (dépendance interne de TRIBE v2).
- Le modèle TRIBE v2 inclut LLaMA 3.2 (modèle gated) — l'accès doit être demandé sur la page HuggingFace de Meta avant le premier téléchargement.
