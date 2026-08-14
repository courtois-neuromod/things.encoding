import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Simulation des données
n_voxels = 1000
n_folds = 5

grille_alphas = np.logspace(1, 20, 20)

# alphas_finaux : vecteur 1D (n_voxels,) — alphas moyens par voxel
alphas_finaux = 10 ** np.random.uniform(1, 20, size=n_voxels)

# alphas_fold : matrice (n_folds, n_voxels) — alphas par fold et par voxel
alphas_fold = 10 ** np.random.uniform(1, 20, size=(n_folds, n_voxels))


def plot_alphas_histogram(alphas_finaux, grille_alphas, alphas_fold=None):
    log10_grille = np.log10(grille_alphas)
    step = log10_grille[1] - log10_grille[0]
    bins = np.append(log10_grille - step / 2, log10_grille[-1] + step / 2)

    # Construction du DataFrame et paramètres spécifiques selon le cas
    if alphas_fold is None:
        log10_valeurs = np.log10(alphas_finaux)
        df = pd.DataFrame({"log10_alpha": log10_valeurs})
        hue_params = {
            "color": "#d73027",
            "kde": True,
            "kde_kws": {"bw_adjust": 0.5},
            "line_kws": {"linewidth": 2},
        }
        titre = "Distribution des alphas moyens"
    else:
        rows = [
            {"log10_alpha": np.log10(v), "fold": f"fold_{i + 1}"}
            for i, fold in enumerate(alphas_fold)
            for v in fold
        ]
        df = pd.DataFrame(rows)
        log10_valeurs = np.log10(alphas_fold.flatten())
        hue_params = {"hue": "fold", "multiple": "stack", "palette": "tab20"}
        titre = "Distribution des alphas par fold"

    # Limites et ticks communs
    xlim_min = log10_valeurs.min() - step / 2
    xlim_max = log10_valeurs.max() + step / 2
    ticks_visibles = log10_grille[
        (log10_grille >= xlim_min) & (log10_grille <= xlim_max)
    ]

    # Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.histplot(data=df, x="log10_alpha", bins=bins, shrink=0.8, ax=ax, **hue_params)
    ax.set_xticks(ticks_visibles)
    ax.set_xticklabels([f"{x:.1f}" for x in ticks_visibles], rotation=45)
    ax.set_xlim(xlim_min, xlim_max)
    ax.set_xlabel("log10(alpha)")
    ax.set_ylabel("Nombre de voxels")
    ax.set_title(titre)
    plt.tight_layout()
    plt.show()
    plt.close()


plot_alphas_histogram(alphas_finaux, grille_alphas, alphas_fold)
