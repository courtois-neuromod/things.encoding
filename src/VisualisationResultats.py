"""Mise en figures des résultats d'encodage produits par `RidgeRegression`.

Séparé de `RidgeRegression.py` : cette classe ne calcule rien, elle reçoit les tableaux
renvoyés par les méthodes `nested_cross_validation_*` et les dessine (cartes cérébrales,
histogrammes, planche PNG récapitulative).

Elle n'importe rien de `RidgeRegression` — les chemins lui sont donnés à la construction
sous la forme d'un objet `CheminsProjet` déjà bâti. La dépendance est donc à sens unique.
"""
from pathlib import Path
import textwrap

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter
from PIL import Image
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.plotting import plot_stat_map
import numpy as np
import pandas as pd
import seaborn as sns

matplotlib.use('Agg')

DPI_FIGURES = 300
SEUIL_AFFICHAGE_BRAIN_MAP = 0.01

# Palette des SCOPES (zones cérébrales comparées). Attention : dans
# `_tracer_barres_accuracy` la couleur code la MÉTRIQUE, ici elle code le scope —
# d'où deux palettes disjointes, pour qu'une couleur ne veuille pas dire deux choses
# d'une figure à l'autre. Suite Okabe-Ito, lisible en daltonisme.
PALETTE_SCOPES = ("#D55E00", "#CC79A7", "#F0E442", "#0072B2", "#009E73")

# Nom complet de chaque ROI, pour que les abréviations des figures soient lisibles sans
# connaître la nomenclature. Les clés correspondent aux noms des datasets du fichier
# ROImask (groupes retinotopy_ROIs / fLoc_ROIs / yeo_ROIs) et aux constantes ROIS_* /
# RESEAUX_PARCELLES_* de RidgeRegression. Une ROI absente d'ici s'affiche sous son seul
# nom, sans description : le fichier ROImask peut contenir plus que ces constantes.
DESCRIPTIONS_ROIS = {
    # retinotopy_ROIs — aires rétinotopiques, définies par cartographie du champ visuel
    "V1": "cortex visuel primaire",
    "V2": "aire visuelle V2",
    "V3": "aire visuelle V3",
    "V3a": "aire V3a — voie dorsale, mouvement",
    "V3b": "aire V3b — voie dorsale",
    "hV4": "quatrième aire visuelle humaine — couleur, forme",
    "VO1": "ventral occipital 1",
    "VO2": "ventral occipital 2",
    "LO1": "lateral occipital 1",
    "LO2": "lateral occipital 2",
    "TO1": "temporo-occipital 1 (≈ hMT) — mouvement",
    "TO2": "temporo-occipital 2 (≈ MST) — mouvement",
    # fLoc_ROIs — aires catégorielles, définies par localizer fonctionnel
    "faceFFA": "fusiform face area — visages",
    "faceOFA": "occipital face area — visages",
    "facepSTS": "sillon temporal supérieur postérieur — visages dynamiques",
    "bodyEBA": "extrastriate body area — corps",
    "scenePPA": "parahippocampal place area — scènes, lieux",
    "sceneOPA": "occipital place area — scènes",
    "sceneMPA": "medial place area (≈ RSC) — scènes",
    # yeo_ROIs — réseaux entiers, et réseaux de l'atlas de parcelles (mode parcelles)
    "visual": "réseau visuel (Yeo-7)",
    "sensorimotor": "réseau sensorimoteur (Yeo-7)",
    "dorsalAttention": "réseau attentionnel dorsal (Yeo-7)",
    "ventralAttention": "réseau attentionnel ventral (Yeo-7)",
    "frontoParietal": "réseau fronto-pariétal (Yeo-7)",
    "defaultMode": "réseau du mode par défaut (Yeo-7)",
    "Vis": "réseau visuel (Schaefer/Yeo-7)",
    "DorsAttn": "réseau attentionnel dorsal (Schaefer/Yeo-7)",
}

# Familles rangées en fin de classement quel que soit leur R² : ce ne sont pas des aires
# du système visuel, les mélanger au tri ferait comparer des objets de nature différente.
FAMILLES_HORS_CLASSEMENT = ("reseau", "autre")


class VisualisationResultats:
    """Produit et sauvegarde toutes les figures d'une analyse d'encodage, pour un sujet
    et une couche donnés."""

    def __init__(self, chemins, subject, layer, flag_precision_voxel):
        """Initialise le contexte d'affichage.

        Args :
            chemins : objet `CheminsProjet` déjà construit (cf.
                `RidgeRegression.get_path_file_by_plateform`). Fournit les chemins de
                l'atlas, de l'anatomie, du fichier ROImask et de la racine de sortie.
            subject : identifiant du sujet, ex. "sub-03" (titres et noms de fichiers).
            layer : nom de la couche TRIBE analysée (titres et noms de fichiers).
            flag_precision_voxel : True = voxels, False = parcelles. Détermine le type
                de masker Nilearn, le vocabulaire des axes et la disponibilité du
                ROImask histogram.
        """
        self.chemins = chemins
        self.subject = subject
        self.layer = layer
        self.flag_precision_voxel = flag_precision_voxel

    def _sauvegarder_figure(self, figure_sauvegardable, nom_fichier, message, **kwargs_savefig):
        """Sauvegarde une figure (matplotlib Figure ou display Nilearn) dans output/.

        Args :
            figure_sauvegardable : objet exposant `.savefig()` (Figure matplotlib ou
                display Nilearn).
            nom_fichier : nom du fichier de sortie (dans output/).
            message : préfixe du message affiché une fois la sauvegarde faite.
            **kwargs_savefig : arguments transmis à `.savefig()`.

        Returns :
            Path : chemin complet du fichier sauvegardé.
        """
        chemin_sortie = self.chemins.root_encoding / "output" / "analysis" / nom_fichier
        chemin_sortie.parent.mkdir(parents=True, exist_ok=True)
        figure_sauvegardable.savefig(chemin_sortie, dpi=DPI_FIGURES, **kwargs_savefig)
        print(f"{message} : {chemin_sortie}")
        return chemin_sortie

    def print_scores(self, scores_finaux, noms_parcelles=None):
        """Affiche un résumé (moyenne, médiane, max, part de R² positifs) des scores R².

        Args :
            scores_finaux : R² par voxel/parcelle.
            noms_parcelles : noms à afficher pour le voxel/parcelle max (sinon son index).
        """
        unite = "voxel" if self.flag_precision_voxel == True else "parcelle"
        index_max = np.argmax(scores_finaux)
        label_max = index_max if noms_parcelles is None else noms_parcelles[index_max]

        print(f"\n=========================================")
        print(f"[Résultats Finaux Robustes — couche {self.layer}]")
        print(f"R² moyen   : {np.mean(scores_finaux):.4f}")
        print(f"R² médian  : {np.median(scores_finaux):.4f}")
        print(f"R² max     : {np.max(scores_finaux):.4f}  ({unite} {label_max})")
        print(f"{unite.capitalize()}s R² > 0 : {np.sum(scores_finaux > 0)} / {len(scores_finaux)}")
        print(f"=========================================")

    def _etendre_valeurs_masque(self, valeurs, masque_roi, valeur_remplissage=0.0):
        """Replace des valeurs calculées sur un sous-ensemble de voxels (ROI) dans un
        vecteur de la taille du cerveau entier, pour l'affichage sur une carte cérébrale.

        Args :
            valeurs : valeurs calculées sur les voxels de la ROI (longueur = nombre
                de voxels True dans `masque_roi`).
            masque_roi : vecteur booléen par voxel, ou None (cerveau entier).
            valeur_remplissage : valeur donnée aux voxels hors ROI.

        Returns :
            np.ndarray : `valeurs` telles quelles si `masque_roi` est None, sinon un
                vecteur de la taille du cerveau entier.
        """
        if masque_roi is None:
            return valeurs
        pleines = np.full(masque_roi.shape[0], valeur_remplissage, dtype=np.float64)
        pleines[masque_roi] = valeurs
        return pleines

    def plot_r2_distribution(self, r2_tous_les_tests, suffix=""):
        """Trace la distribution du R² moyen (inter-folds) par voxel/parcelle et
        l'enregistre en PNG.

        Args :
            r2_tous_les_tests : R² par fold externe et par voxel/parcelle, shape
                (n_folds, n_features).
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        r2_moyen = np.mean(r2_tous_les_tests, axis=0)
        mediane = np.median(r2_moyen)
        unite = "voxels" if self.flag_precision_voxel else "parcelles"

        df_moyen = pd.DataFrame({"r2": r2_moyen})

        fig, ax = plt.subplots(figsize=(10, 5))

        sns.histplot(
            data=df_moyen, x="r2",
            bins=100, element="step", fill=False,
            linewidth=2, color="black",
            label=f"{self.subject} (med {mediane:.3f})",
            ax=ax,
        )

        ax.axvline(mediane, color="black", linestyle="--", linewidth=1)
        ax.axvline(0, color="grey", linestyle=":", linewidth=1)
        ax.set_xlabel("per-voxel R² (raw)")
        ax.set_ylabel(f"Nombre de {unite}")
        ax.set_title(f"Per-voxel R² distribution — {self.subject} / {self.layer} ({unite})")
        ax.legend()
        plt.tight_layout()

        nom_fichier = f"r2_distribution_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Distribution R² sauvegardée")
        plt.close(fig)
        return chemin_sauvegarde

    def _ordonner_rois_par_famille(self, df):
        """Calcule l'ordre d'affichage des ROIs et les bornes des blocs de famille.

        Familles classées par R² moyen décroissant, ROIs décroissantes à l'intérieur de
        chaque famille. Les familles de `FAMILLES_HORS_CLASSEMENT` sont renvoyées en fin
        de liste quel que soit leur R².

        Args :
            df : DataFrame avec les colonnes "ROI", "r2_mean" et "famille".

        Returns :
            tuple : (ordre des ROIs, [(nom de famille, index de fin de bloc), ...]).
                L'index de fin est exclu, à la façon d'une tranche Python.
        """
        r2_par_famille = df.groupby("famille")["r2_mean"].mean().sort_values(ascending=False)

        familles_classees = [f for f in r2_par_famille.index if f not in FAMILLES_HORS_CLASSEMENT]
        familles_classees += [f for f in FAMILLES_HORS_CLASSEMENT if f in r2_par_famille.index]

        ordre, bornes = [], []
        for famille in familles_classees:
            rois_famille = df[df["famille"] == famille].sort_values("r2_mean", ascending=False)
            ordre.extend(rois_famille["ROI"].tolist())
            bornes.append((famille, len(ordre)))
        return ordre, bornes

    def _tracer_barres_roi(self, ax, df, colonne, ordre, bornes, palette, label_x, noms_familles):
        """Trace un panneau de barres horizontales par ROI sur l'axe donné.

        Args :
            ax : axe matplotlib cible.
            df : DataFrame "ROI" / "famille" / colonnes de scores.
            colonne : colonne de score à tracer ("r2_mean" ou "pearson_mean").
            ordre : ordre des ROIs, identique sur tous les panneaux (cf.
                `_ordonner_rois_par_famille`) pour que la lecture se fasse ligne à ligne.
            bornes : bornes des blocs de famille, pour les séparateurs.
            palette : couleur par famille.
            label_x : légende de l'axe X.
            noms_familles : True pour écrire le nom des familles en marge (panneau de
                gauche uniquement, sinon le texte se répète inutilement).
        """
        sns.barplot(
            data=df, y="ROI", x=colonne, order=ordre,
            hue="famille", palette=palette, dodge=False, ax=ax, legend=False,
        )

        # Le R² comme le Pearson peuvent être négatifs (voxels non prédits) : le zéro
        # n'est pas forcément au bord du cadre, on le matérialise.
        ax.axvline(0, color="grey", linewidth=0.8, alpha=0.6, zorder=0)

        # Pas de barre d'erreur ici (une seule valeur par ROI), donc `bar_label` place
        # correctement les étiquettes tout seul, y compris pour les barres négatives.
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
        ax.margins(x=0.20)

        # Séparateurs entre familles + nom de la famille en marge du bloc.
        debut = 0
        for famille, fin in bornes:
            if fin < len(ordre):
                ax.axhline(fin - 0.5, color="grey", linewidth=0.9, alpha=0.45)
            if noms_familles:
                ax.text(
                    -0.34, (debut + fin - 1) / 2, famille,
                    transform=ax.get_yaxis_transform(),
                    ha="center", va="center", rotation=90,
                    fontsize=9, fontweight="bold", color=palette.get(famille, "black"),
                )
            debut = fin

        ax.grid(True, axis='x', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel(label_x, fontsize=11)
        ax.set_ylabel("")

    def _texte_descriptions_rois(self, ordre, bornes, noms_courts):
        """Compose le bloc de texte qui explicite les abréviations des ROIs tracées.

        Args :
            ordre : étiquettes des ROIs dans l'ordre d'affichage du graphique.
            bornes : bornes des blocs de famille (même ordre que le graphique).
            noms_courts : {étiquette affichée: nom brut de la ROI}, le nom brut étant
                la clé de `DESCRIPTIONS_ROIS`.

        Returns :
            str : texte monospace, deux ROIs par ligne, groupé par famille.
        """
        largeur_roi = max(len(noms_courts[roi]) for roi in ordre) + 1
        entree = lambda roi: (
            f"{noms_courts[roi]:<{largeur_roi}} {DESCRIPTIONS_ROIS.get(noms_courts[roi], '')}".rstrip()
        )
        # Largeur de colonne calculée sur TOUTES les entrées, pas famille par famille :
        # sinon chaque bloc aurait sa propre colonne et le texte paraîtrait décalé.
        largeur_colonne = max(len(entree(roi)) for roi in ordre) + 3

        lignes = []
        debut = 0
        for famille, fin in bornes:
            entrees = [entree(roi) for roi in ordre[debut:fin]]
            debut = fin
            for i in range(0, len(entrees), 2):
                corps = "".join(e.ljust(largeur_colonne) for e in entrees[i:i + 2]).rstrip()
                prefixe = f"{famille:<9}" if i == 0 else " " * 9
                lignes.append(prefixe + corps)
        return "\n".join(lignes)

    def plot_ROImask_histogram(self, scores_finaux, pearson_finaux=None):
        """Trace le score moyen par ROI (voxelwise uniquement) et l'enregistre en PNG.

        Les ROIs sont groupées par famille (blocs de couleur contigus), les familles
        classées par R² décroissant et les ROIs décroissantes à l'intérieur de chaque
        famille. Chaque barre porte sa valeur, et un bloc de texte sous la figure
        explicite les abréviations.

        Args :
            scores_finaux : R² moyen par voxel (cerveau entier, même longueur que les
                masques du fichier ROImask).
            pearson_finaux : corrélations de Pearson au même format, si la méthode de
                validation croisée en produit (`one_cycle` seule) ; None = figure à un
                seul panneau.

        Returns :
            Path | None : chemin du PNG sauvegardé, ou None si `flag_precision_voxel`
                est False (analyse indisponible en précision parcelle).

        Notes :
            L'ordre des ROIs est calculé UNE fois, sur le R², et réutilisé tel quel sur
            le panneau Pearson (`sharey=True`) : les deux panneaux se lisent alors ligne
            à ligne, ce qui fait ressortir les ROIs où le Pearson est bon alors que le R²
            reste faible — celles dont la forme temporelle est prédite mais pas
            l'amplitude. Trier chaque panneau séparément casserait cette lecture.
        """
        fichier_ROImask = self.chemins.chemin_ROImask

        # Chaque ROI est rattachée à une famille, colorée via `palette` ci-dessous.
        # Les clés doivent correspondre EXACTEMENT aux noms des datasets du fichier
        # ROImask (groupes retinotopy_ROIs / fLoc_ROIs / yeo_ROIs) : la boucle plus bas
        # parcourt TOUT le fichier, pas seulement les ROIs listées dans les constantes
        # de RidgeRegression. Toute ROI absente d'ici retombe sur "autre", qui doit donc
        # exister dans `palette` sous peine de faire lever seaborn.
        familles = {
            # retinotopy_ROIs
            "V1": "early", "V2": "early", "V3": "early",
            "hV4": "ventral", "VO1": "ventral", "VO2": "ventral",
            "V3a": "ventral", "V3b": "ventral",
            "LO1": "lateral", "LO2": "lateral", "TO1": "lateral", "TO2": "lateral",
            # fLoc_ROIs
            "faceFFA": "face", "faceOFA": "face", "facepSTS": "face",
            "scenePPA": "scene", "sceneOPA": "scene", "sceneMPA": "scene",
            "bodyEBA": "body",
            # yeo_ROIs : réseaux entiers, pas des ROIs visuelles — famille à part pour
            # ne pas les faire passer pour des aires du système visuel.
            "visual": "reseau", "sensorimotor": "reseau", "dorsalAttention": "reseau",
            "ventralAttention": "reseau", "frontoParietal": "reseau", "defaultMode": "reseau",
        }
        palette = {
            "early": "#1f77b4",
            "ventral": "#ff7f0e",
            "lateral": "#2ca02c",
            "face": "#d62728",
            "scene": "#e377c2",
            "body": "#8c564b",
            "reseau": "#7f7f7f",
            "autre": "#bdbdbd",
        }

        if not self.flag_precision_voxel:
            print("ROImask ignoré : analyse disponible uniquement en précision voxel (flag_precision_voxel=True).")
            return None

        rows = []
        with h5py.File(fichier_ROImask, 'r') as fichier:
            for groupe in fichier.keys():
                for sous_cle in fichier[groupe].keys():
                    vecteur = fichier[groupe][sous_cle][:].astype(bool)
                    ligne = {
                        # L'effectif est dans l'étiquette : une moyenne sur 97 voxels et
                        # une sur 2 024 ne se lisent pas de la même façon.
                        "ROI": f"{sous_cle} (n={int(vecteur.sum()):,})".replace(",", " "),
                        "r2_mean": np.mean(scores_finaux[vecteur]),
                        "famille": familles.get(sous_cle, "autre"),
                        "nom_court": sous_cle,
                    }
                    if pearson_finaux is not None:
                        ligne["pearson_mean"] = np.mean(pearson_finaux[vecteur])
                    rows.append(ligne)

        df = pd.DataFrame(rows)
        ordre, bornes = self._ordonner_rois_par_famille(df)

        if pearson_finaux is None:
            fig, ax_r2 = plt.subplots(figsize=(9, 11))
        else:
            # sharey : le panneau Pearson reprend l'ordre calculé sur le R², et les noms
            # de ROIs ne sont écrits qu'une fois, à gauche.
            fig, (ax_r2, ax_pearson) = plt.subplots(1, 2, figsize=(15, 11), sharey=True)

        self._tracer_barres_roi(ax_r2, df, "r2_mean", ordre, bornes, palette,
                                "R² moyen (raw)", noms_familles=True)
        if pearson_finaux is not None:
            self._tracer_barres_roi(ax_pearson, df, "pearson_mean", ordre, bornes, palette,
                                    "Pearson r moyen", noms_familles=False)

        fig.suptitle(
            f"Encodage par ROI visuelle — {self.subject} / {self.layer}",
            fontsize=14, fontweight="bold",
        )
        # Bande réservée en haut pour la légende, qui se glisse sous le titre.
        fig.tight_layout(rect=(0, 0, 1, 0.945))

        # Légende des couleurs commune aux deux panneaux (seaborn ne la trace plus,
        # `legend=False`). En haut plutôt qu'en bas : le bas est occupé par le bloc de
        # descriptions, et les deux se chevaucheraient.
        familles_tracees = [famille for famille, _ in bornes]
        fig.legend(
            handles=[Patch(facecolor=palette[f], label=f) for f in familles_tracees],
            loc="upper center", bbox_to_anchor=(0.5, 0.962),
            ncol=len(familles_tracees), frameon=False, fontsize=10,
        )

        # Bloc qui explicite les abréviations, sous la figure : `bbox_inches="tight"` à
        # la sauvegarde englobe ce qui dépasse. `ma="left"` est indispensable — sans lui
        # chaque ligne serait centrée indépendamment et les colonnes ne seraient plus
        # alignées.
        noms_courts = dict(zip(df["ROI"], df["nom_court"]))
        fig.text(
            0.5, -0.015, self._texte_descriptions_rois(ordre, bornes, noms_courts),
            ha="center", ma="left", va="top", fontsize=8, family="monospace", linespacing=1.5,
        )

        nom_fichier = f"ROImask_{self.subject}_{self.layer}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "ROImask sauvegardé", bbox_inches="tight")
        plt.close(fig)
        return chemin_sauvegarde

    def plot_r2_threshold(self, r2_tous_les_tests, suffix=""):
        """Trace la fraction de voxels/parcelles dont le R² moyen dépasse un seuil,
        pour une grille de seuils entre 0 et le R² max, et l'enregistre en PNG.

        Args :
            r2_tous_les_tests : R² par fold externe et par voxel/parcelle, shape
                (n_folds, n_features).
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        r2_moyen = np.mean(r2_tous_les_tests, axis=0)
        unite = "voxels" if self.flag_precision_voxel else "parcelles"

        max_r2 = r2_moyen.max() if r2_moyen.max() > 0 else 0.3
        seuils = np.linspace(0.0, max_r2, 300)
        fractions = [np.mean(r2_moyen >= seuil) for seuil in seuils]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(True, axis='both', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)

        ax.plot(seuils, fractions, linewidth=2.5, color="#0072B2", label=self.subject)
        ax.axvline(0.05, color="grey", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(0.10, color="grey", linestyle="--", linewidth=1, alpha=0.5)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("R² threshold", fontsize=12)
        ax.set_ylabel(f"fraction of {unite} ≥ threshold", fontsize=12)
        ax.set_title(f"How many {unite} are well predicted", fontsize=14, fontweight='bold')
        ax.legend(frameon=False, loc="upper right")

        plt.tight_layout()

        nom_fichier = f"r2_threshold_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Threshold R² sauvegardé")
        plt.close(fig)
        return chemin_sauvegarde

    def _stats_par_fold(self, scores_par_fold):
        """Construit le tableau long (une ligne par fold × métrique) attendu par
        `sns.barplot` pour tracer mean / median / top-10% / max.

        Une valeur PAR FOLD externe (et pas une seule valeur agrégée par métrique) :
        c'est ce qui permet à `errorbar="sd"` de tracer l'écart-type inter-folds.

        Args :
            scores_par_fold : scores par fold externe et par voxel/parcelle, shape
                (n_folds, n_features).

        Returns :
            pd.DataFrame : colonnes "Métrique" et "Score".
        """
        n_folds = scores_par_fold.shape[0]

        means_par_fold = np.mean(scores_par_fold, axis=1)
        medians_par_fold = np.median(scores_par_fold, axis=1)
        seuils_top10_par_fold = np.percentile(scores_par_fold, 90, axis=1)
        top10_par_fold = np.array([
            np.mean(fold[fold >= seuil])
            for fold, seuil in zip(scores_par_fold, seuils_top10_par_fold)
        ])
        # Le meilleur voxel/parcelle du fold : borne haute de ce que le modèle atteint
        # là où il marche le mieux, que la moyenne du top-10% lisse déjà.
        max_par_fold = np.max(scores_par_fold, axis=1)

        return pd.DataFrame({
            "Métrique": (["mean"] * n_folds + ["median"] * n_folds
                         + ["top-10% mean"] * n_folds + ["max"] * n_folds),
            "Score": np.concatenate([means_par_fold, medians_par_fold, top10_par_fold, max_par_fold]),
        })

    def _tracer_barres_accuracy(self, ax, scores_par_fold, label_y):
        """Trace un panneau de barres mean/median/top-10%/max sur l'axe donné."""
        ax.grid(True, axis='both', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)

        df = self._stats_par_fold(scores_par_fold)
        ordre_metriques = ["mean", "median", "top-10% mean", "max"]

        # Palette Okabe-Ito (sûre pour les daltonismes) : bleu, bleu clair, orange, vert.
        palette = {"mean": "#0072B2", "median": "#56B4E9", "top-10% mean": "#E69F00", "max": "#009E73"}
        sns.barplot(
            data=df, x="Métrique", y="Score", hue="Métrique",
            palette=palette, ax=ax, errorbar="sd", capsize=0.1, legend=False,
        )

        # seaborn ne renseigne pas `container.errorbar`, donc `ax.bar_label` collerait
        # les étiquettes au sommet des barres, par-dessus les moustaches. On les place
        # nous-mêmes au-dessus de la moustache haute (même écart-type que seaborn,
        # ddof=1 ; NaN si un seul fold).
        ecarts = df.groupby("Métrique")["Score"].std().reindex(ordre_metriques).fillna(0.0)
        for nom_metrique, container in zip(ordre_metriques, ax.containers):
            barre = container[0]
            hauteur = barre.get_height()
            ax.annotate(
                f"{hauteur:.3f}",
                xy=(barre.get_x() + barre.get_width() / 2, hauteur + ecarts[nom_metrique]),
                xytext=(0, 3), textcoords="offset points",
                ha='center', va='bottom', fontsize=9,
            )
        ax.margins(y=0.12)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel(label_y, fontsize=12)

    def plot_accuracy(self, r2_tous_les_tests, pearson_tous_les_tests=None, suffix=""):
        """Trace les barres mean/median/top-10%/max (moyenne ± écart-type inter-folds)
        pour UN sujet et les enregistre en PNG. Pas d'agrégation multi-sujets : un
        seul appel = un seul sujet, une seule figure.

        Args :
            r2_tous_les_tests : R² par fold externe et par voxel/parcelle, shape
                (n_folds, n_features).
            pearson_tous_les_tests : corrélations de Pearson au même format, si la
                méthode de validation croisée en produit (seule `one_cycle`
                aujourd'hui) ; None = figure à un seul panneau.
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path : chemin du fichier PNG sauvegardé.

        Notes :
            Les deux panneaux ont des échelles indépendantes : le Pearson ignore le
            biais et l'échelle de la prédiction, donc il est mécaniquement plus haut
            que le R², et les forcer sur un axe commun écraserait les barres R².
        """
        n_features = r2_tous_les_tests.shape[1]
        n_folds = r2_tous_les_tests.shape[0]

        if pearson_tous_les_tests is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            self._tracer_barres_accuracy(ax, r2_tous_les_tests, "R² (raw)")
            # Titre plus petit sur un seul panneau : contrairement à un titre d'axe,
            # un suptitle n'est pas rétréci par tight_layout et déborderait des 6 pouces.
            taille_titre = 11
        else:
            fig, (ax_r2, ax_pearson) = plt.subplots(1, 2, figsize=(12, 6))
            self._tracer_barres_accuracy(ax_r2, r2_tous_les_tests, "R² (raw)")
            self._tracer_barres_accuracy(ax_pearson, pearson_tous_les_tests, "Pearson r")
            taille_titre = 14

        fig.suptitle(
            f"Accuracy — {self.subject} / {self.layer} (n={n_features:,}, {n_folds} folds)",
            fontsize=taille_titre, fontweight='bold',
        )
        plt.tight_layout()

        nom_fichier = f"accuracy_{self.subject}_{self.layer}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Accuracy sauvegardée")
        plt.close(fig)
        return chemin_sauvegarde


    def _tracer_barres_comparaison(self, ax, scores_par_scope, label_y):
        """Trace un panneau de barres mean/median/top-10%/max groupées par scope."""
        ax.grid(True, axis='both', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)

        # `_stats_par_fold` est réutilisé tel quel : une ligne par (métrique × fold),
        # ce qui laisse `errorbar="sd"` calculer l'écart-type inter-folds exactement
        # comme dans `plot_accuracy`. On n'ajoute que la colonne qui sépare les scopes.
        frames, etiquettes = [], {}
        for nom_scope, scores in scores_par_scope.items():
            df_scope = self._stats_par_fold(scores)
            # L'effectif est dans la légende : les scopes vont de ~180 à 1134 unités,
            # et cet écart change la lecture des barres (surtout celle du max).
            etiquettes[nom_scope] = f"{nom_scope} (n={scores.shape[1]:,})"
            df_scope["Scope"] = etiquettes[nom_scope]
            frames.append(df_scope)
        df = pd.concat(frames, ignore_index=True)

        ordre_metriques = ["mean", "median", "top-10% mean", "max"]
        ordre_scopes = [etiquettes[nom] for nom in scores_par_scope]
        palette = {
            etiquette: PALETTE_SCOPES[i % len(PALETTE_SCOPES)]
            for i, etiquette in enumerate(ordre_scopes)
        }

        sns.barplot(
            data=df, x="Métrique", y="Score", hue="Scope",
            order=ordre_metriques, hue_order=ordre_scopes,
            palette=palette, ax=ax, errorbar="sd", capsize=0.06,
        )

        # Même problème que dans `_tracer_barres_accuracy` : seaborn ne renseigne pas
        # `container.errorbar`, donc on place les étiquettes au-dessus de la moustache
        # haute. Différence ici : un container est un SCOPE et contient une barre par
        # métrique (l'autre helper, avec hue == x, n'en a qu'une), d'où la double
        # boucle et l'écart-type pris sur le couple (scope, métrique).
        ecarts = df.groupby(["Scope", "Métrique"])["Score"].std().fillna(0.0)
        for etiquette, container in zip(ordre_scopes, ax.containers):
            for nom_metrique, barre in zip(ordre_metriques, container):
                hauteur = barre.get_height()
                ax.annotate(
                    f"{hauteur:.3f}",
                    xy=(barre.get_x() + barre.get_width() / 2,
                        hauteur + ecarts[(etiquette, nom_metrique)]),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=7, rotation=90,
                )
        # Plus de marge que sur la figure à un scope : les étiquettes sont verticales.
        ax.margins(y=0.22)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel(label_y, fontsize=12)
        ax.legend(title="", fontsize=9, frameon=False)

    def _texte_composition_scopes(self, composition_scopes, scopes_traces, largeur=150):
        """Compose le bloc de texte qui détaille le contenu de chaque scope comparé.

        Args :
            composition_scopes : {nom du scope: itérable de noms de ROIs/réseaux}.
            scopes_traces : noms des scopes réellement présents dans la figure, dans
                l'ordre d'affichage.
            largeur : largeur de repli, en caractères.

        Returns :
            str : un paragraphe par scope, vide si rien à décrire.
        """
        paragraphes = []
        for nom_scope in scopes_traces:
            noms = composition_scopes.get(nom_scope)
            # Le scope « tout le cerveau » n'a pas de composition à lister.
            if not noms:
                continue
            details = ", ".join(
                f"{nom} ({DESCRIPTIONS_ROIS[nom]})" if nom in DESCRIPTIONS_ROIS else nom
                for nom in noms
            )
            entete = f"{nom_scope} ({len(tuple(noms))}) — "
            paragraphes.append(textwrap.fill(
                entete + details, width=largeur,
                subsequent_indent=" " * 4,
            ))
        return "\n".join(paragraphes)

    def plot_comparaison_scopes(self, nom_methode, r2_par_scope, pearson_par_scope=None,
                                composition_scopes=None, suffix=""):
        """Compare les zones cérébrales analysées dans UNE figure, pour une méthode.

        Remplace les planches séparées par scope : mêmes métriques que `plot_accuracy`,
        mais une couleur par scope au lieu d'une figure par scope. Les scopes n'étant
        que des sous-ensembles de colonnes d'une même CV, les valeurs affichées sont
        celles d'un unique passage (cf. `ResultatsCV.restreindre`).

        Args :
            nom_methode : nom de la méthode de validation croisée (titre de la figure).
            r2_par_scope : {nom du scope: R² par fold, shape (n_folds, n_features)}.
                Le nombre de features diffère d'un scope à l'autre, c'est attendu.
            pearson_par_scope : mêmes clés, corrélations de Pearson, si la méthode en
                produit (`one_cycle` seule) ; None = figure à un seul panneau.
            composition_scopes : {nom du scope: noms des ROIs/réseaux qui le composent},
                détaillé en bloc de texte sous les panneaux. Les noms connus de
                `DESCRIPTIONS_ROIS` sont accompagnés de leur définition. Les clés
                absentes ne sont pas décrites — c'est le cas du scope « tout le
                cerveau », qui n'a pas de composition à lister.
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path | None : chemin du PNG sauvegardé, ou None si aucun scope à comparer.

        Notes :
            - `max` est la seule des quatre barres qui ne se compare pas à effectif
              égal : un grand scope contient les unités d'un petit, donc son maximum
              lui est mécaniquement supérieur ou égal. Les trois autres sont des
              moyennes, insensibles à l'effectif.
            - Échelles indépendantes entre les deux panneaux, pour la raison déjà
              documentée dans `plot_accuracy`.
        """
        if not r2_par_scope:
            print(f"Aucun scope à comparer pour {nom_methode}.")
            return None

        n_folds = next(iter(r2_par_scope.values())).shape[0]

        if pearson_par_scope is None:
            fig, ax = plt.subplots(figsize=(8, 6))
            self._tracer_barres_comparaison(ax, r2_par_scope, "R² (raw)")
            taille_titre = 11
        else:
            fig, (ax_r2, ax_pearson) = plt.subplots(1, 2, figsize=(15, 6))
            self._tracer_barres_comparaison(ax_r2, r2_par_scope, "R² (raw)")
            self._tracer_barres_comparaison(ax_pearson, pearson_par_scope, "Pearson r")
            taille_titre = 14

        unite = "voxels analysés" if self.flag_precision_voxel else "parcelles analysées"
        fig.suptitle(
            f"Comparaison des {unite} — {nom_methode} — "
            f"{self.subject} / {self.layer} ({n_folds} folds)",
            fontsize=taille_titre, fontweight='bold',
        )
        plt.tight_layout()

        # Contenu de chaque scope sous les panneaux : sans ça, un scope nommé "ROIs" ou
        # "visuelles" ne dit pas quelles aires il recouvre. `bbox_inches="tight"`
        # englobe ce qui dépasse ; `ma="left"` garde l'indentation du repli.
        if composition_scopes:
            texte = self._texte_composition_scopes(composition_scopes, list(r2_par_scope))
            if texte:
                fig.text(
                    0.5, -0.02, texte,
                    ha="center", ma="left", va="top", fontsize=8,
                    family="monospace", linespacing=1.5,
                )

        nom_fichier = f"comparaison_scopes_{self.subject}_{self.layer}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(
            fig, nom_fichier, "Comparaison des scopes sauvegardée", bbox_inches="tight",
        )
        plt.close(fig)
        return chemin_sauvegarde

    def plot_alphas_histogram(self, alphas_fold, grille_alphas, alphas_finaux=None, titre=None, suffix=""):
        """Trace la distribution (log10) des alphas sélectionnés et l'enregistre en PNG.

        Args :
            alphas_fold : alphas par fold externe et par voxel/parcelle (ignoré si
                `alphas_finaux` est fourni). Une couleur par ligne du tableau, donc à
                n'utiliser que si les lignes SONT bien des folds externes.
            grille_alphas : grille complète d'alphas testés (fixe les bins/ticks).
            alphas_finaux : si fourni, affiche une distribution unique (une courbe)
                plutôt que la distribution empilée par fold. Accepte n'importe quel
                tableau d'alphas aplati.
            titre : titre de la figure ; None = titre par défaut selon la branche.
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        log10_grille = np.log10(grille_alphas)
        step = log10_grille[1] - log10_grille[0]
        bins = np.append(log10_grille - step / 2, log10_grille[-1] + step / 2)

        if alphas_finaux is not None:
            log10_valeurs = np.log10(alphas_finaux)
            df = pd.DataFrame({"log10_alpha": log10_valeurs})
            hue_params = {"color": "#d73027", "kde": True, "kde_kws": {"bw_adjust": 0.5}, "line_kws": {"linewidth": 2}}
            titre_defaut = "Distribution des alphas moyens"
        else:
            alphas_fold = np.array(alphas_fold)
            rows = [{"log10_alpha": np.log10(v), "fold": f"fold_{i + 1}"}
                    for i, fold in enumerate(alphas_fold) for v in fold]
            df = pd.DataFrame(rows)
            log10_valeurs = np.log10(alphas_fold.flatten())
            hue_params = {"hue": "fold", "multiple": "dodge", "palette": "tab20"}
            titre_defaut = "Distribution des alphas par fold"

        xlim_min = log10_valeurs.min() - step / 2
        xlim_max = log10_valeurs.max() + step / 2
        ticks_visibles = log10_grille[(log10_grille >= xlim_min) & (log10_grille <= xlim_max)]

        unite = "voxels" if self.flag_precision_voxel else "parcelles"
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x="log10_alpha", bins=bins, shrink=0.8, ax=ax, **hue_params)
        ax.set_xticks(ticks_visibles)
        ax.set_xticklabels([f"{x:.1f}" for x in ticks_visibles], rotation=45)
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_xlabel("log10(alpha)")
        ax.set_ylabel(f"Nombre de {unite}")
        ax.set_title(titre if titre is not None else titre_defaut)
        plt.tight_layout()
        nom_fichier = f"histogram_alphas_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Histogramme alphas sauvegardé")
        plt.close(fig)
        return chemin_sauvegarde

    def _brain_mapping_generique(self, donnees, nom_carte, cmap, treshold=SEUIL_AFFICHAGE_BRAIN_MAP, echelle_log=False, vmin=None, vmax=None, suffix=""):
        """Projette un vecteur de scores (R², alphas, TSNR...) sur le cerveau et
        enregistre la carte statistique en PNG.

        Args :
            donnees : un score par voxel/parcelle (même longueur que le masque de
                l'atlas du sujet).
            nom_carte : nom affiché dans le titre et le nom de fichier (ex. "R2").
            cmap : colormap matplotlib.
            treshold : seuil d'affichage transmis à `plot_stat_map` (valeurs
                masquées si |valeur affichée| <= treshold).
            echelle_log : si True, affiche log10(donnees) (colorbar reformatée en
                conséquence).
            vmin, vmax : bornes de la colorbar (None = calculées par Nilearn).
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        chemins = self.chemins

        donnees_affichees = np.log10(donnees) if echelle_log else donnees
        coords_R2_map = {'x': np.array([-52.5, -28.5, -12.5, 9.5, 21.5, 35.5, 47.5]), 'y': np.array([-96.5, -80.5, -60.5, -42.5, -26.5, 53.5, 69.5]), 'z': np.array([-18.5, -4.5, 7.5, 19.5, 31.5, 45.5, 61.5])}

        if self.flag_precision_voxel == True:
            masker = NiftiMasker(mask_img=chemins.chemin_atlas, standardize=False)
            kwargs = {"bg_img": chemins.chemin_anatomie}
        else:
            masker = NiftiLabelsMasker(labels_img=chemins.chemin_atlas, standardize=False)
            kwargs = {"cut_coords": coords_R2_map}

        masker.fit()
        r2_map_3d = masker.inverse_transform(donnees_affichees)

        fig = plt.figure(figsize=(14, 10), facecolor='white')

        display = plot_stat_map(
            r2_map_3d,
            figure=fig,
            threshold=treshold,
            vmin=vmin,
            vmax=vmax,
            symmetric_cbar=False,
            display_mode='mosaic',
            cbar_tick_format="%.2f",
            colorbar=True,
            cmap=cmap,
            **kwargs,
        )
        unite = "voxel" if self.flag_precision_voxel == True else "parcelle"
        title=f'{nom_carte} pour {self.subject} - {self.layer} en {unite}'
        fig.suptitle(title, fontsize=18, fontweight='bold', color='black', y=0.98, ha='center')
        fig.subplots_adjust(top=0.92)

        if echelle_log and display._cbar is not None:
            display._cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda valeur, position: f"$10^{{{valeur:.0f}}}$"))

        nom_fichier = f"brain_map_{self.subject}_{self.layer}_{nom_carte}_{unite}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(display, nom_fichier, "Carte cérébrale sauvegardée")
        display.close()
        plt.close(fig)
        return chemin_sauvegarde

    def brain_mapping_r2(self, scores_r2, noms_parcelles=None, suffix="", masque_roi=None):
        """Affiche le résumé des R² (`print_scores`) et enregistre la carte cérébrale
        correspondante.

        Args :
            scores_r2 : R² par voxel/parcelle. Si `masque_roi` est fourni, ne porte
                que sur les voxels de la ROI (les stats de `print_scores` restent
                alors calculées sur la ROI seule).
            noms_parcelles : transmis à `print_scores` pour nommer le score max.
            suffix : suffixe ajouté au nom du fichier de sortie.
            masque_roi : vecteur booléen par voxel ; si fourni, la carte est projetée
                sur le cerveau entier, les voxels hors ROI étant remplis à 0 (donc
                masqués par `SEUIL_AFFICHAGE_BRAIN_MAP`).

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        self.print_scores(scores_r2, noms_parcelles)
        donnees_carte = self._etendre_valeurs_masque(scores_r2, masque_roi, valeur_remplissage=0.0)
        chemin_sauvegarde = self._brain_mapping_generique(donnees_carte, nom_carte="R2", cmap="YlOrRd", treshold=SEUIL_AFFICHAGE_BRAIN_MAP, echelle_log=False, vmin=0, vmax=np.max(scores_r2), suffix=suffix)
        return chemin_sauvegarde

    def brain_mapping_alphas(self, alphas_tous_les_lots, suffix="", masque_roi=None):
        """Enregistre la carte cérébrale des alphas optimaux (échelle log10).

        Args :
            alphas_tous_les_lots : alpha optimal par voxel/parcelle.
            suffix : suffixe ajouté au nom du fichier de sortie.
            masque_roi : vecteur booléen par voxel ; si fourni, les voxels hors ROI
                sont remplis à 1.0 (donc log10(1)=0, masqué par le seuil d'affichage
                — cohérent avec le traitement du R² dans `brain_mapping_r2`).

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        donnees_carte = self._etendre_valeurs_masque(alphas_tous_les_lots, masque_roi, valeur_remplissage=1.0)
        chemin_sauvegarde = self._brain_mapping_generique(donnees_carte, nom_carte="Alphas", cmap="YlOrRd", treshold=0, echelle_log=True, suffix=suffix)
        return chemin_sauvegarde

    def brain_mapping_tsnr(self, tsnr, suffix=""):
        """Enregistre la carte cérébrale du TSNR (borné au 95e percentile pour éviter
        que les valeurs extrêmes n'écrasent la colorbar).

        Args :
            tsnr : TSNR par voxel/parcelle.
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        chemin_sauvegarde = self._brain_mapping_generique(tsnr, nom_carte="TSNR", cmap="Blues", treshold=0.0, echelle_log=False, vmin=0, vmax=np.percentile(tsnr, 95), suffix=suffix)
        return chemin_sauvegarde

    def regrouper_figures_dans_une_planche(self, nom_methode, liste_chemins_figures,
                                           nombre_de_colonnes=3, figures_pleine_largeur=()):
        """Assemble toutes les figures PNG déjà sauvegardées pour UNE méthode de
        validation croisée dans une seule image PNG (une "planche"), au lieu d'avoir
        un fichier séparé par figure.

        Args :
            nom_methode : nom de la méthode (utilisé dans le titre et le nom de fichier).
            liste_chemins_figures : chemins renvoyés par les appels à `brain_mapping_r2`,
                `brain_mapping_alphas`, `plot_accuracy`, etc. Les entrées None (ex.
                ROImask quand `flag_precision_voxel` est False) sont ignorées.
            nombre_de_colonnes : largeur de la grille d'assemblage.
            figures_pleine_largeur : chemins qui occupent une rangée entière au lieu
                d'une case. À réserver aux figures larges (ROImask à deux panneaux,
                comparaison des scopes) : tassées dans une case carrée, leurs étiquettes
                deviendraient illisibles.

        Returns :
            Path | None : chemin du fichier PNG sauvegardé, ou None si
                `liste_chemins_figures` ne contient aucune figure valide.
        """
        # Étape 1 : on ne garde que les chemins réellement produits (pas les None).
        chemins_valides = [c for c in liste_chemins_figures if c is not None]
        if not chemins_valides:
            print(f"Aucune figure à regrouper pour {nom_methode}.")
            return None

        pleine_largeur = {c for c in figures_pleine_largeur if c is not None}

        # Étape 2 : on répartit les figures en rangées. Une figure pleine largeur ferme
        # la rangée en cours (même incomplète) et occupe seule la sienne.
        rangees = []
        rangee_courante = []
        for chemin_figure in chemins_valides:
            if chemin_figure in pleine_largeur:
                if rangee_courante:
                    rangees.append(rangee_courante)
                    rangee_courante = []
                rangees.append([chemin_figure])
            else:
                rangee_courante.append(chemin_figure)
                if len(rangee_courante) == nombre_de_colonnes:
                    rangees.append(rangee_courante)
                    rangee_courante = []
        if rangee_courante:
            rangees.append(rangee_courante)

        # Étape 3 : hauteur de chaque rangée. Une rangée pleine largeur est dimensionnée
        # d'après le rapport de forme de son image (lu sans la décoder), pour ne pas
        # l'écraser — plafonné, sinon un ROImask très haut ferait exploser la planche.
        largeur_par_case = 6
        hauteur_par_case = 6
        largeur_totale = largeur_par_case * nombre_de_colonnes

        proportions_hauteur = []
        for rangee in rangees:
            if len(rangee) == 1 and rangee[0] in pleine_largeur:
                largeur_px, hauteur_px = Image.open(rangee[0]).size
                hauteur_voulue = largeur_totale * hauteur_px / largeur_px
                proportions_hauteur.append(min(hauteur_voulue, 3 * hauteur_par_case) / hauteur_par_case)
            else:
                proportions_hauteur.append(1.0)

        hauteur_totale = hauteur_par_case * sum(proportions_hauteur)
        figure_planche = plt.figure(figsize=(largeur_totale, hauteur_totale))
        grille = figure_planche.add_gridspec(
            len(rangees), nombre_de_colonnes, height_ratios=proportions_hauteur,
        )

        # Étape 4 : on place chaque image dans sa case (ou sur toute la rangée).
        for numero_ligne, rangee in enumerate(rangees):
            if len(rangee) == 1 and rangee[0] in pleine_largeur:
                cases = [figure_planche.add_subplot(grille[numero_ligne, :])]
            else:
                cases = [
                    figure_planche.add_subplot(grille[numero_ligne, numero_colonne])
                    for numero_colonne in range(nombre_de_colonnes)
                ]

            for case, chemin_image in zip(cases, rangee):
                case.imshow(plt.imread(chemin_image))
                case.set_title(Path(chemin_image).stem, fontsize=8)

            # On masque toujours les axes (case vide ou pas) pour un rendu propre.
            for case in cases:
                case.axis("off")

        # Bande réservée au titre : hauteur fixée en POUCES puis convertie, car `y` et
        # `rect` sont en fraction de figure et la planche n'a pas de hauteur fixe. Avec
        # les valeurs par défaut, le titre retomberait sur la première rangée dès que la
        # planche s'allonge.
        bande_titre_pouces = 0.6
        titre_planche = f"{nom_methode} — {self.subject} / {self.layer}"
        figure_planche.suptitle(
            titre_planche, fontsize=16, fontweight="bold",
            y=1 - 0.15 / hauteur_totale, va="top",
        )
        figure_planche.tight_layout(rect=(0, 0, 1, 1 - bande_titre_pouces / hauteur_totale))

        nom_fichier_planche = f"planche_{nom_methode}_{self.subject}_{self.layer}.png"
        chemin_sauvegarde = self._sauvegarder_figure(figure_planche, nom_fichier_planche, "Planche de figures sauvegardée")
        plt.close(figure_planche)

        return chemin_sauvegarde

    def generer_toutes_les_figures(self, nom_methode, resultats, grille_alphas, noms_parcelles=None,
                                   masque_roi=None, r2_par_scope=None, pearson_par_scope=None,
                                   composition_scopes=None):
        """Génère et sauvegarde l'ensemble des figures standard pour UNE méthode de
        validation croisée (appelée une fois par méthode : full_manuel, ridgecv_loo,
        one_cycle...).

        Args :
            nom_methode : nom de la méthode (utilisé dans les noms de fichiers).
            resultats : objet renvoyé par la méthode `nested_cross_validation_*`
                correspondante (une `ResultatsCV`, cf. `RidgeRegression`). Ses champs
                sont lus par attribut, sans import : la dépendance reste à sens unique.
                Les champs optionnels (`best_alphas_inner` pour `full_manuel`, les
                champs Pearson pour `one_cycle`) valent None ailleurs, et les figures
                correspondantes sont alors simplement sautées.
            grille_alphas : grille d'alphas testée (transmise aux histogrammes).
            noms_parcelles : transmis à `brain_mapping_r2`/`print_scores`.
            masque_roi : vecteur booléen par voxel ; si fourni, les cartes cérébrales
                projettent les valeurs sur le cerveau entier (voxels hors ROI
                masqués), et le ROImask histogram (déjà une ventilation par ROI) est
                sauté puisque l'analyse est déjà restreinte à une ROI.
            r2_par_scope, pearson_par_scope, composition_scopes : transmis tels quels à
                `plot_comparaison_scopes`. Fournir `r2_par_scope` ajoute la comparaison
                des zones cérébrales À LA PLANCHE, plutôt que dans un fichier à part :
                une méthode = un seul PNG. None = pas de comparaison.

        Returns :
            dict : résumé ("r2_moyen", "r2_tous_les_tests", "alphas_moyens",
                "pearson_moyen") réutilisable pour comparer les méthodes entre elles
                dans `__main__`.
        """
        r2_moyen = resultats.r2_moyen
        r2_variance_inter_folds = resultats.r2_variance_inter_folds
        r2_tous_les_tests = resultats.r2_tous_les_tests
        alphas_tous_externes = resultats.alphas_tous_externes
        best_alphas_inner = resultats.best_alphas_inner
        pearson_tous_les_tests = resultats.pearson_tous_les_tests
        tsnr = resultats.TSNR

        suffix = f"_{nom_methode}"
        # Moyenne géométrique sur les folds (les alphas s'étalent sur plusieurs décades)
        alphas_moyens = 10 ** np.mean(np.log10(alphas_tous_externes), axis=0)

        print(f"\n[FIGURES] {nom_methode} — Variance inter-folds moyenne : {np.mean(r2_variance_inter_folds):.6f}")
        if resultats.pearson_moyen is not None:
            print(f"[FIGURES] {nom_methode} — Pearson moyen : {np.mean(resultats.pearson_moyen):.4f}")

        # Liste explicite des chemins de chaque figure produite pour cette méthode :
        # elle sert ensuite à assembler toutes ces figures dans une seule planche.
        liste_chemins_figures = []

        # 1. brain_r2_map + 2. alpha_map (+ TSNR, cohérent avec les deux autres cartes)
        print(" -> Cartes cérébrales (R², Alphas, TSNR)...")
        chemin_figure = self.brain_mapping_r2(r2_moyen, noms_parcelles, suffix=suffix, masque_roi=masque_roi)
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.brain_mapping_alphas(alphas_moyens, suffix=suffix, masque_roi=masque_roi)
        liste_chemins_figures.append(chemin_figure)

        #chemin_figure = self.brain_mapping_tsnr(tsnr, suffix=suffix)
        #liste_chemins_figures.append(chemin_figure)

        # 3. histogrammes des alphas par fold + 4. moyenne des alphas
        print(" -> Histogrammes des alphas...")
        chemin_figure = self.plot_alphas_histogram(alphas_fold=alphas_tous_externes, grille_alphas=grille_alphas, suffix=f"{suffix}_folds")
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.plot_alphas_histogram(alphas_fold=None, grille_alphas=grille_alphas, alphas_finaux=alphas_moyens, suffix=f"{suffix}_moyen")
        liste_chemins_figures.append(chemin_figure)

        # 5. visualisation des alphas internes (uniquement si la méthode en produit).
        # `best_alphas_inner` contient une entrée par (fold externe × session interne),
        # pas une par fold externe : les colorer "fold_1", "fold_2"... serait faux, et
        # illisible (plusieurs centaines de séries). On trace donc la distribution
        # globale de tous les alphas internes retenus.
        if best_alphas_inner is not None:
            chemin_figure = self.plot_alphas_histogram(
                alphas_fold=None,
                grille_alphas=grille_alphas,
                alphas_finaux=np.asarray(best_alphas_inner).reshape(-1),
                titre="Distribution des alphas des folds internes (LOGO, toutes sessions × folds externes)",
                suffix=f"{suffix}_inner",
            )
            liste_chemins_figures.append(chemin_figure)
        else:
            print(f"  -> Pas d'alphas internes pour {nom_methode} (pas de sous-CV interne).")

        # 6. courbe d'accuracy (single-subject), distributionr2 et r2 treshold.
        # Le Pearson, quand la méthode en produit, s'affiche en second panneau de la
        # figure accuracy — pas en carte cérébrale : il dessinerait la même topographie
        # que le R², sur une échelle différente.
        print(" -> Accuracy et distribution R²...")
        chemin_figure = self.plot_accuracy(r2_tous_les_tests, pearson_tous_les_tests=pearson_tous_les_tests, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.plot_r2_distribution(r2_tous_les_tests, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.plot_r2_threshold(r2_tous_les_tests, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        # Figures larges, qui prennent une rangée entière de la planche au lieu d'une
        # case : tassées dans un carré, leurs étiquettes deviendraient illisibles.
        figures_pleine_largeur = []

        # ROIMask (uniquement en précision voxel, et uniquement pour l'analyse cerveau
        # entier : sur une analyse déjà restreinte à une ROI, la ventilation par ROI
        # n'apporte rien de plus)
        if self.flag_precision_voxel and masque_roi is None:
            print(" -> ROIMask...")
            chemin_figure = self.plot_ROImask_histogram(r2_moyen, pearson_finaux=resultats.pearson_moyen)
            liste_chemins_figures.append(chemin_figure)
            figures_pleine_largeur.append(chemin_figure)
        elif masque_roi is None:
            print(f"  -> ROIMask ignoré pour {nom_methode} (nécessite flag_precision_voxel=True).")

        # Comparaison des zones cérébrales : produite ici plutôt que dans un fichier à
        # part, pour qu'une méthode ne laisse qu'un seul PNG derrière elle.
        if r2_par_scope:
            print(" -> Comparaison des scopes...")
            chemin_figure = self.plot_comparaison_scopes(
                nom_methode, r2_par_scope, pearson_par_scope,
                composition_scopes=composition_scopes, suffix=suffix,
            )
            liste_chemins_figures.append(chemin_figure)
            figures_pleine_largeur.append(chemin_figure)

        # Regroupement de toutes les figures ci-dessus dans un seul fichier PNG.
        print(" -> Assemblage de la planche de figures...")
        chemin_planche = self.regrouper_figures_dans_une_planche(
            nom_methode, liste_chemins_figures, figures_pleine_largeur=figures_pleine_largeur,
        )

        # Une fois la planche assemblée, les figures individuelles ne servent plus :
        # on les supprime pour ne garder que le fichier fusionné.
        if chemin_planche is not None:
            for chemin_figure in liste_chemins_figures:
                if chemin_figure is not None and chemin_figure != chemin_planche:
                    chemin_figure.unlink(missing_ok=True)
            print(f" -> Figures individuelles supprimées ({len(liste_chemins_figures)} fichiers).")

        return {
            "r2_moyen": r2_moyen,
            "r2_tous_les_tests": r2_tous_les_tests,
            "alphas_moyens": alphas_moyens,
            "pearson_moyen": resultats.pearson_moyen,
        }
