"""
Régression Ridge pour l'encodage cérébral THINGS memory.
Entraîne une RidgeCV par couche et évalue la prédiction.
"""
from pathlib import Path

from GroupShuffleSplitSession import GroupShuffleSplitSession
from TribeHDF5Normalization import TribeHDF5Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
from sklearn.compose import TransformedTargetRegressor
from sklearn.multioutput import MultiOutputRegressor
import numpy as np
import pandas as pd
import h5py
import gc
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.plotting import plot_stat_map
import matplotlib
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from dataclasses import dataclass
import seaborn as sns
import warnings
from scipy.linalg import LinAlgWarning

# Ignore spécifiquement les avertissements de matrices mal conditionnées
warnings.filterwarnings(action='ignore', category=LinAlgWarning)

matplotlib.use('Agg')

@dataclass
class CheminsProjet:
    """Regroupe tous les chemins de fichiers nécessaires pour un sujet donné."""

    root_encoding: Path
    root_timeseries: Path
    chemin_tribe: Path
    chemin_cneuromod: Path
    chemin_atlas: Path
    chemin_ROImask: Path
    chemin_anatomie: Path = None


class RidgeRegression:
    """Entraîne et évalue une régression Ridge pour prédire l'activité IRMf
    à partir des activations d'une couche du modèle TRIBE, par sujet."""

    def __init__(self, plateforme, subject, layer,  flag_delai_bold_brute, centrage_donne_temps, flag_precision_voxel, ROImask_flag, randomize_flag = False):
        self.plateforme = plateforme
        self.subject = subject
        self.layer = layer
        self.flag_delai_bold_brute = flag_delai_bold_brute
        self.centrage_donne_temps = centrage_donne_temps
        self.flag_precision_voxel = flag_precision_voxel
        self.randomize_flag = randomize_flag
        self.ROImask_flag = ROImask_flag


    def get_path_file_by_plateform(self, plateforme):
        """Construit les chemins de fichiers du sujet selon la plateforme
        (cluster Rorqual ou poste local) et le niveau de précision (voxel/parcelle)."""
        if plateforme == "Rorqual":
            ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
            ROOT_TIMESERIES = Path("/home/aclaud/links/scratch/things.timeseries")
        else:
            ROOT_ENCODING = Path(__file__).parent.parent
            ROOT_TIMESERIES = ROOT_ENCODING

        chemin_tribe = ROOT_ENCODING / "output" / "hdf5" / "things_encoding" / f"{self.subject}.h5"
        chemin_ROImask = ROOT_ENCODING / "data" / "brain_map_subj" / f"{self.subject}_space-T1w_desc-ROImasks_voxelAnnotations.h5"

        if self.flag_precision_voxel:
            sous_dossier = ROOT_TIMESERIES / "timeseries" / "voxel_native" / self.subject
            chemin_cneuromod = sous_dossier / f"{self.subject}_task-things_space-T1w_desc-voxelwise_timeseries.h5"
            chemin_atlas = sous_dossier / f"{self.subject}_task-things_space-T1w_label-GMfromFS_desc-indivFunc_mask.nii.gz"
            chemin_anatomie = sous_dossier / f"{self.subject}_desc-preproc_T1w.nii.gz"
        else:
            sous_dossier = ROOT_TIMESERIES / "timeseries" / "cneuromod2026" / self.subject
            chemin_cneuromod = sous_dossier / f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
            chemin_atlas = sous_dossier / f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_dseg.nii.gz"
            chemin_anatomie = None

        return CheminsProjet(ROOT_ENCODING, ROOT_TIMESERIES, chemin_tribe, chemin_cneuromod, chemin_atlas, chemin_ROImask, chemin_anatomie)

    def get_chemin_annotations_parcelles(self, plateforme):
        """Retourne le chemin du fichier TSV contenant les noms des parcelles de l'atlas."""
        nom_fichier_annotations = (
            "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018TianS3NettekovenAsym_"
            "desc-1000Parcels7Networks50Subcort128Cereb_parcelAnnotations.tsv"
        )
        if plateforme == "Rorqual":
            ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
            return ROOT_ENCODING / "data" / "brain_map_subj" / nom_fichier_annotations
        else:
            ROOT = Path(__file__).parent.parent
            return ROOT / "data" / "brain_map_subj" / nom_fichier_annotations

    def charger_noms_parcelles(self, plateforme):
        """Charge la liste des noms de parcelles depuis le fichier d'annotations."""
        chemin_annotations = self.get_chemin_annotations_parcelles(plateforme)
        annotations = pd.read_csv(chemin_annotations, sep="\t")
        return annotations["name"].tolist()

    def discover_runs(self,tribe_hdf5=None):
        """Liste les runs disponibles dans le fichier HF5 contenant les embeddings TRIBE et fait correspondre
        chacun à sa session/run CNeuroMod et à sa vidéo source."""
        chemins = self.get_path_file_by_plateform(self.plateforme)

        if not chemins.chemin_tribe.exists():
            print(f"Erreur : Fichier introuvable : {chemins.chemin_tribe}")

        runs = []
        gere_localement = tribe_hdf5 is None

        if gere_localement:
            tribe_hdf5 = h5py.File(chemins.chemin_tribe, "r")

        try:
            for tribe_ses in sorted(tribe_hdf5.keys()):           # "ses-001", "ses-002", ...
                for tribe_run in sorted(tribe_hdf5[tribe_ses].keys()):   # "run-1", "run-2", ...

                    # Conversion ses-001 → ses-01 pour CNeuroMod
                    num_ses = int(tribe_ses.replace("ses-", ""))
                    cneuromod_ses = f"ses-{num_ses:02d}"

                    # Clé dataset CNeuroMod
                    num_run = tribe_run.replace("run-", "")
                    cneuromod_dataset = f"{cneuromod_ses}_task-things_run-{num_run}_timeseries"

                    # Chemin vidéo originale (non CFR) pour ffprobe
                    nom_video = f"{self.subject}_{tribe_ses}_task-thingsmemory_{tribe_run}.mp4"
                    if self.plateforme == "Rorqual":
                        chemin_video = chemins.root_encoding / "data" / "data" / self.subject / tribe_ses / nom_video
                    else:
                        chemin_video = chemins.root_encoding / "data" / self.subject / tribe_ses / nom_video

                    runs.append((tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset))
        finally:
            if gere_localement:
                tribe_hdf5.close()

        print(f"{len(runs)} runs trouvés dans {chemins.chemin_tribe.name}")
        return runs

    def create_X_Y_total(self):
        """Construit les matrices X (activations) et Y (signal IRMf) en alignant
        temporellement chaque run, puis les concatène sur l'ensemble des runs.

        Returns:
            tuple: (runs_ok, X, Y, groupes, TSNR)
                - runs_ok : liste des runs traités avec succès.
                - X, Y : matrices concaténées.
                - groupes : identifiant de session pour chaque échantillon.
                - TSNR : rapport signal/bruit temporel par voxel/parcelle.
        """
        chemins = self.get_path_file_by_plateform(self.plateforme)

        # Alignement temporel et concaténation
        X_list, Y_list = [], []
        runs_ok = []
        groupes_list = []

        with h5py.File(chemins.chemin_tribe, "r") as tribe_hdf5, \
                h5py.File(chemins.chemin_cneuromod, "r") as cneuromod_hdf5:

            runs = self.discover_runs(tribe_hdf5=tribe_hdf5)

            for (tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset) in runs:
                # Vérifier que la vidéo source existe localement
                if self.subject == "sub-06" and cneuromod_ses == "ses-08" and tribe_run == "run-06" :
                    print(f"{cneuromod_ses} ignorée pour {self.subject} (décision manuelle : mauvais alignement).")
                    continue

                if not chemin_video.exists():
                    print(f"Vidéo manquante, run ignoré : {chemin_video.name}")
                    continue

                if cneuromod_ses not in cneuromod_hdf5 or cneuromod_dataset not in cneuromod_hdf5[cneuromod_ses]:
                    print(f"CNeuroMod : Données IRMf absentes pour {cneuromod_ses} / {cneuromod_dataset}. Run ignoré.")
                    continue

                normalisateur = TribeHDF5Normalization(
                    chemin_tribe=chemins.chemin_tribe,
                    chemin_cneuromod=chemins.chemin_cneuromod,
                    chemin_video=chemin_video,
                    tribe_ses=tribe_ses,
                    tribe_run=tribe_run,
                    tribe_layer=self.layer,
                    cneuromod_ses=cneuromod_ses,
                    cneuromod_dataset=cneuromod_dataset,
                    t_Tribe_s=0.5,
                    TR_irmf_s=1.49,
                    flag_delai_bold_brute=self.flag_delai_bold_brute,
                    centrage_donne_temps=self.centrage_donne_temps,
                )
                X_run, Y_run = normalisateur.executer_pipeline(
                    tribe_hdf5=tribe_hdf5, cneuromod_hdf5=cneuromod_hdf5
                )
                X_list.append(X_run)
                Y_list.append(Y_run)

                runs_ok.append(f"{tribe_ses}/{tribe_run}")
                num_ses = int(tribe_ses.replace("ses-", ""))
                id_array = np.full(X_run.shape[0], num_ses)
                groupes_list.append(id_array)

            print(f"\n{len(runs_ok)} runs traités avec succès")

            if self.randomize_flag:
                rng = np.random.default_rng(42)
                nombre_de_runs = len(Y_list)

                while True:
                    nouvel_ordre = rng.permutation(nombre_de_runs)
                    if not np.any(nouvel_ordre == np.arange(nombre_de_runs)):
                        break

                Y_list = [Y_list[i] for i in nouvel_ordre]
                print(f"⚠ Baseline activée : Y_list réordonné aléatoirement ({nombre_de_runs} runs, aucun n'a gardé sa position d'origine)")

            X = np.concatenate(X_list, axis=0)
            Y = np.concatenate(Y_list, axis=0)

            TSNR = Y.mean(axis=0) / (Y.std(axis=0) + 1e-8)

            groupes = np.concatenate(groupes_list, axis=0)
            print(f"Matrice finale : X={X.shape}, Y={Y.shape}")

            del X_list, Y_list, groupes_list

            return runs_ok, X, Y, groupes, TSNR

    def _selection_X_Y(self, sessions_a_exclure=None):
        """Construit X, Y et exclut, si demandé, les sessions données."""
        runs_ok, X, Y, groupes, TSNR = self.create_X_Y_total()
        if sessions_a_exclure is not None:
            masque = ~np.isin(groupes, sessions_a_exclure)
            X, Y, groupes = X[masque], Y[masque], groupes[masque]
        return X, Y, groupes, TSNR

    def _scaler_X_Y(self, X, Y, train_mask, test_mask):
        """Standardise X et Y : le scaler est ajusté sur le train uniquement,
        puis appliqué au train et au test pour éviter toute fuite de données."""
        X_train, X_test = X[train_mask], X[test_mask]
        Y_train, Y_test = Y[train_mask], Y[test_mask]

        scaler_X = StandardScaler()
        X_scaled_train = scaler_X.fit_transform(X_train)
        X_scaled_test = scaler_X.transform(X_test)

        scaler_Y = StandardScaler()
        Y_scaled_train = scaler_Y.fit_transform(Y_train)
        Y_scaled_test = scaler_Y.transform(Y_test)
        return X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test

    def _ridge_par_lots(self, X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test,alphas, taille_lot, n_folds=None, index_fold=None):
        """Entraîne une RidgeCV par lots de features (voxels/parcelles) pour
        limiter l'empreinte mémoire, et retourne le R² et l'alpha optimal par feature.
        """
        n_features = Y_scaled_train.shape[1]
        n_lots = int(np.ceil(n_features / taille_lot))

        r2_lots = np.zeros(n_features, dtype=np.float32)
        alphas_lots = np.zeros(n_features, dtype=np.float64)

        for index_lot, debut in enumerate(range(0, n_features, taille_lot)):

            fin = min(debut + taille_lot, n_features)

            if alphas.shape[0] == n_features:
                grille_alphas_lot = np.unique(alphas[debut:fin])
            else:
                grille_alphas_lot = alphas

            # Boucle interne avec LOO analytique
            modele = RidgeCV(
                alphas=grille_alphas_lot,
                alpha_per_target=True,
                cv=None,  # LOO activé
                fit_intercept=True,
            )

            modele.fit(X_scaled_train, Y_scaled_train[:, debut:fin])

            # Evaluation sur le fold test
            Y_pred = modele.predict(X_scaled_test)

            r2_lots[debut:fin] = r2_score(Y_scaled_test[:, debut:fin], Y_pred, multioutput="raw_values")
            alphas_lots[debut:fin] = modele.alpha_
            del modele, Y_pred
            gc.collect()

        return r2_lots, alphas_lots

    def nested_cross_validation(self, grille_alphas, n_folds=5, test_size=0.2, seed=None):
        """Validation croisée imbriquée 100% manuelle (Méthode de la Moyenne Géométrique)."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import LeaveOneGroupOut
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score
        import gc

        X, Y, groupes, TSNR = self._selection_X_Y()
        n_features = Y.shape[1]

        # 1. Définition du splitter externe
        outer_cv = GroupShuffleSplitSession(n_splits=n_folds, test_size=test_size, random_state=seed)

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)

        # 2. BOUCLE EXTERNE : Évaluation de la stabilité du modèle
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y, groupes)):
            print(f"  -> Début du Fold externe {i + 1}/{n_folds}...")

            X_train, Y_train, groupes_train = X[train_idx], Y[train_idx], groupes[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            inner_cv = LeaveOneGroupOut()
            inner_splits = list(inner_cv.split(X_train, Y_train, groups=groupes_train))

            n_inner_folds = len(inner_splits)
            best_alphas_inner = np.zeros((n_inner_folds, n_features), dtype=np.float64)

            # On teste chaque fold interne (une session isolée en validation)
            for j, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):

                # Standardisation locale stricte au fold interne (0 fuite)
                scaler_X_inner = StandardScaler()
                X_inner_train_scaled = scaler_X_inner.fit_transform(X_train[inner_train_idx])
                X_inner_val_scaled = scaler_X_inner.transform(X_train[inner_val_idx])

                scaler_Y_inner = StandardScaler()
                Y_inner_train_scaled = scaler_Y_inner.fit_transform(Y_train[inner_train_idx])
                Y_inner_val_scaled = scaler_Y_inner.transform(Y_train[inner_val_idx])

                # Tableau pour stocker les R² de chaque alpha
                r2_par_alpha = np.zeros((len(grille_alphas), n_features))

                # On teste chaque alpha de la grille explicitement
                for a_idx, alpha in enumerate(grille_alphas):
                    # Un seul modèle Ridge pour tout le cerveau
                    ridge_inner = Ridge(alpha=alpha)
                    ridge_inner.fit(X_inner_train_scaled, Y_inner_train_scaled)
                    Y_inner_pred = ridge_inner.predict(X_inner_val_scaled)

                    # On stocke les performances de cet alpha pour toutes les parcelles
                    r2_par_alpha[a_idx, :] = r2_score(Y_inner_val_scaled, Y_inner_pred, multioutput='raw_values')

                # Pour chaque voxel, on cherche l'index de l'alpha qui a maximisé le R²
                best_indices = np.argmax(r2_par_alpha, axis=0)
                best_alphas_inner[j, :] = grille_alphas[best_indices]

            # Moyenne géométrique = exp(mean(log(valeurs)))
            alphas_moyens_geom = np.exp(np.mean(np.log(best_alphas_inner), axis=0))
            alphas_tous_externes[i, :] = alphas_moyens_geom

            # Standardisation du set externe
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            scaler_Y = StandardScaler()
            Y_train_scaled = scaler_Y.fit_transform(Y_train)
            Y_test_scaled = scaler_Y.transform(Y_test)

            ridge_final = Ridge(alpha=alphas_moyens_geom)
            ridge_final.fit(X_train_scaled, Y_train_scaled)

            # Évaluation sur le test set
            Y_pred_scaled = ridge_final.predict(X_test_scaled)

            # Calcul du score R2
            r2_score_fold = r2_score(Y_test_scaled, Y_pred_scaled, multioutput='raw_values')
            r2_tous_les_tests[i, :] = r2_score_fold
            print(f"-> R2 mean : {np.mean(r2_score_fold)}")
            print(f"-> R2 max : {np.max(r2_score_fold)}")

            # Nettoyage mémoire
            del ridge_final, Y_pred_scaled
            gc.collect()

        # 3. Calcul des métriques finales
        r2_moyen = np.mean(r2_tous_les_tests, axis=0)
        r2_variance_inter_folds = np.var(r2_tous_les_tests, axis=0)
        alphas_tous_externes_moyen = np.mean(alphas_tous_externes, axis=0)

        return r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, TSNR

    def print_scores(self, scores_finaux, noms_parcelles=None):
        """Affiche un résumé (moyenne, médiane, max, part de R² positifs) des scores R²."""
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

    def plot_r2_distribution(self, r2_tous_les_tests, suffix=""):
        r2_moyen = np.mean(r2_tous_les_tests, axis=0)  # (n_voxels,)
        mediane = np.median(r2_moyen)
        unite = "voxels" if self.flag_precision_voxel else "parcelles"

        # DataFrame long format : une ligne par (fold, voxel)
        rows = []
        for i in range(r2_tous_les_tests.shape[0]):
            for j in range(r2_tous_les_tests.shape[1]):
                rows.append({"fold": i, "r2": r2_tous_les_tests[i, j]})
        df_folds = pd.DataFrame(rows)

        # DataFrame pour la moyenne inter-folds
        df_moyen = pd.DataFrame({"r2": r2_moyen})

        fig, ax = plt.subplots(figsize=(10, 5))

        # Distribution moyenne inter-folds (au premier plan)
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

        chemins = self.get_path_file_by_plateform(self.plateforme)
        nom_fichier = f"r2_distribution_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sortie = chemins.root_encoding / "output" / nom_fichier
        plt.savefig(chemin_sortie, dpi=300)
        plt.close()
        print(f"Distribution R² sauvegardée : {chemin_sortie}")

    def plot_ROImask_histogram(self, scores_finaux):
        """Trace un boxplot des R² par ROI (voxelwise uniquement) et l'enregistre en HTML."""
        chemins = self.get_path_file_by_plateform(self.plateforme)
        fichier_ROImask = chemins.chemin_ROImask

        # Mapping ROI → famille de couleur
        familles = {
            "V1": "early", "V2": "early", "V3": "early",
            "hv4": "ventral", "V3a": "ventral", "V3b": "ventral",
            "faceFFA": "face", "faceOFA": "face", "faceSTS": "face",
            "scenePPA": "scene", "sceneOPA": "scene", "sceneMPA": "scene",
            "bodyEBA": "body",
            "dorsalAttention": "lateral", "ventralAttention": "lateral",
        }
        palette = {
            "early": "#1f77b4",
            "ventral": "#ff7f0e",
            "lateral": "#2ca02c",
            "face": "#d62728",
            "scene": "#e377c2",
            "body": "#8c564b",
        }

        if self.flag_precision_voxel:
            rows = []
            with h5py.File(fichier_ROImask, 'r') as fichier:
                for groupe in fichier.keys():
                    for sous_cle in fichier[groupe].keys():
                        vecteur = fichier[groupe][sous_cle][:].astype(bool)
                        r2_roi = scores_finaux[vecteur]
                        rows.append({
                            "ROI": sous_cle,
                            "r2_mean": np.mean(r2_roi),
                            "famille": familles.get(sous_cle, "autre"),
                        })

                df = pd.DataFrame(rows).sort_values("r2_mean", ascending=True)
                print(df.shape)

                fig, ax = plt.subplots(figsize=(8, 10))
                sns.barplot(
                    data=df, y="ROI", x="r2_mean",
                    hue="famille", palette=palette,
                    dodge=False, ax=ax
                )
                ax.set_xlabel("mean R² (raw)")
                ax.set_ylabel("")
                ax.set_title(f"Encoding accuracy by visual ROI — {self.subject} / {self.layer}")
                ax.legend(title="stream", bbox_to_anchor=(1.05, 1), loc="upper left")
                plt.tight_layout()

                nom_fichier = f"ROImask_{self.subject}_{self.layer}.png"
                plt.savefig(chemins.root_encoding / "output" / nom_fichier, dpi=300, bbox_inches="tight")
                plt.close()
        else:
            return "Pas en voxel"

    def plot_r2_threshold(self, r2_tous_les_tests, suffix=""):
        r2_moyen = np.mean(r2_tous_les_tests, axis=0)  # (n_voxels,)
        unite = "voxels" if self.flag_precision_voxel else "parcelles"

        # On démarre à 0 (comme sur l'image) jusqu'au R2 maximum
        max_r2 = r2_moyen.max() if r2_moyen.max() > 0 else 0.3
        seuils = np.linspace(0.0, max_r2, 300)
        fractions = [np.mean(r2_moyen >= seuil) for seuil in seuils]

        fig, ax = plt.subplots(figsize=(8, 6))

        # Ajout de la grille légère en arrière-plan
        ax.grid(True, axis='both', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)

        # Tracé de la courbe
        ax.plot(seuils, fractions, linewidth=2.5, color="#0072B2", label=self.subject)

        # Lignes verticales de repère (transparentes et pointillées)
        ax.axvline(0.05, color="grey", linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(0.10, color="grey", linestyle="--", linewidth=1, alpha=0.5)

        # Nettoyage de l'esthétique (retrait des bordures haut et droite)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("R² threshold", fontsize=12)
        ax.set_ylabel(f"fraction of {unite} ≥ threshold", fontsize=12)

        # Titre centré et en gras
        ax.set_title(f"How many {unite} are well predicted", fontsize=14, fontweight='bold')

        # Légende (en haut à droite, sans cadre)
        ax.legend(frameon=False, loc="upper right")

        plt.tight_layout()

        chemins = self.get_path_file_by_plateform(self.plateforme)
        nom_fichier = f"r2_threshold_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sortie = chemins.root_encoding / "output" / nom_fichier
        plt.savefig(chemin_sortie, dpi=300)
        plt.close()
        print(f"Threshold R² sauvegardé : {chemin_sortie}")

    def plot_accuracy(self, r2_tous_les_tests):
        r2_moyen = np.mean(r2_tous_les_tests, axis=0)  # (n_voxels ou n_parcelles,)
        n_features = r2_moyen.shape[0]

        mean = np.mean(r2_moyen)
        median = np.median(r2_moyen)
        seuil_top10 = np.percentile(r2_moyen, 90)
        top10 = np.mean(r2_moyen[r2_moyen >= seuil_top10])

        # Création du label pour l'axe X (Sujet + N)
        x_label = f"{self.subject}\n(n={n_features:,})"

        # Restructuration du DataFrame pour utiliser le paramètre 'hue' de Seaborn
        df = pd.DataFrame({
            "Sujet": [x_label, x_label, x_label],
            "Métrique": ["mean", "median", "top-10% mean"],
            "R2": [mean, median, top10],
        })

        fig, ax = plt.subplots(figsize=(6, 6))

        # Ajout de la grille légère en arrière-plan (comme sur l'image)
        ax.grid(True, axis='both', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)  # S'assure que la grille reste derrière les barres

        # Couleurs fidèles à votre image de référence
        palette = {"mean": "#0072B2", "median": "#56B4E9", "top-10% mean": "#E69F00"}

        # Le 'hue' crée la légende automatiquement et groupe les barres
        sns.barplot(
            data=df, x="Sujet", y="R2", hue="Métrique",
            palette=palette, ax=ax
        )

        # Annotation automatique et propre des valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

        # Nettoyage de l'esthétique (retrait des bordures haut et droite)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("")  # On laisse vide car le label du tick suffit
        ax.set_ylabel("R² (raw)", fontsize=12)

        # Titre centré et en gras
        ax.set_title(f"Per-subject accuracy — {self.subject} / {self.layer}", fontsize=14, fontweight='bold')

        # Positionnement de la légende (sans cadre, en haut à gauche)
        ax.legend(title="", frameon=False, loc="upper left")

        plt.tight_layout()

        chemins = self.get_path_file_by_plateform(self.plateforme)
        nom_fichier = f"accuracy_{self.subject}_{self.layer}.png"
        chemin_sortie = chemins.root_encoding / "output" / nom_fichier
        plt.savefig(chemin_sortie, dpi=300)
        plt.close()
        print(f"Accuracy sauvegardée : {chemin_sortie}")


    def plot_alphas_histogram(self, alphas_fold, grille_alphas, alphas_finaux=None, suffix=""):
        """Trace la distribution (log10) des alphas sélectionnés et l'enregistre en PNG.

        Si `alphas_finaux` est fourni, affiche la distribution des alphas moyens
        (une courbe). Sinon, affiche la distribution empilée par fold à partir
        de `alphas_fold`.
        """
        log10_grille = np.log10(grille_alphas)
        step = log10_grille[1] - log10_grille[0]
        bins = np.append(log10_grille - step / 2, log10_grille[-1] + step / 2)

        # Construction du DataFrame et paramètres spécifiques selon le cas
        if alphas_finaux is not None:
            log10_valeurs = np.log10(alphas_finaux)
            df = pd.DataFrame({"log10_alpha": log10_valeurs})
            hue_params = {"color": "#d73027", "kde": True, "kde_kws": {"bw_adjust": 0.5}, "line_kws": {"linewidth": 2}}
            titre = "Distribution des alphas moyens"
        else:
            alphas_fold = np.array(alphas_fold)
            rows = [{"log10_alpha": np.log10(v), "fold": f"fold_{i + 1}"}
                    for i, fold in enumerate(alphas_fold) for v in fold]
            df = pd.DataFrame(rows)
            log10_valeurs = np.log10(alphas_fold.flatten())
            hue_params = {"hue": "fold", "multiple": "dodge", "palette": "tab20"}
            titre = "Distribution des alphas par fold"

        # Limites et ticks communs
        xlim_min = log10_valeurs.min() - step / 2
        xlim_max = log10_valeurs.max() + step / 2
        ticks_visibles = log10_grille[(log10_grille >= xlim_min) & (log10_grille <= xlim_max)]

        # Figure
        unite = "voxels" if self.flag_precision_voxel else "parcelles"
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data=df, x="log10_alpha", bins=bins, shrink=0.8, ax=ax, **hue_params)
        ax.set_xticks(ticks_visibles)
        ax.set_xticklabels([f"{x:.1f}" for x in ticks_visibles], rotation=45)
        ax.set_xlim(xlim_min, xlim_max)
        ax.set_xlabel("log10(alpha)")
        ax.set_ylabel(f"Nombre de {unite}")
        ax.set_title(titre)
        plt.tight_layout()
        nom_fichier = f"histogram_alphas_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sortie = self.get_path_file_by_plateform(self.plateforme).root_encoding / "output" / nom_fichier
        plt.savefig(chemin_sortie, dpi=300)
        plt.close()
        print(f"Histogramme alphas sauvegardé : {chemin_sortie}")

    def _brain_mapping_generique(self, donnees, nom_carte, cmap, treshold = 0.01, echelle_log=False, vmin = None, vmax = None, suffix=""):
        """Projette un vecteur de scores (R², alphas, TSNR...) sur le cerveau et enregistre la carte statistique en PNG."""
        chemins = self.get_path_file_by_plateform(self.plateforme)

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

        chemin_sortie = chemins.root_encoding / "output" / f"brain_map_{self.subject}_{self.layer}_{nom_carte}_{unite}{suffix}.png"
        display.savefig(chemin_sortie, dpi=300)
        display.close()
        plt.close(fig)
        print(f"Carte cérébrale sauvegardée : {chemin_sortie}")
        return

    def brain_mapping_r2(self, scores_r2, noms_parcelles=None, suffix=""):
        """Affiche le résumé des R² et enregistre la carte cérébrale correspondante."""
        self.print_scores(scores_r2, noms_parcelles)
        self._brain_mapping_generique(scores_r2, nom_carte="R2", cmap="YlOrRd", treshold=0.01, echelle_log=False, vmin=0, vmax=np.max(scores_r2), suffix=suffix)

    def brain_mapping_alphas(self, alphas_tous_les_lots, suffix=""):
        """Enregistre la carte cérébrale des alphas optimaux (échelle log10)."""
        self._brain_mapping_generique(alphas_tous_les_lots, nom_carte="Alphas", cmap="YlOrRd", treshold=0.01, echelle_log=True, suffix=suffix)

    def brain_mapping_tsnr(self, tsnr, suffix=""):
        """Enregistre la carte cérébrale correspondante."""
        # évite que les valeurs extrêmes écrasent la colorbar
        self._brain_mapping_generique(tsnr, nom_carte="TSNR", cmap="Blues", treshold=0.0, echelle_log=False,vmin=0,vmax=np.percentile(tsnr, 95),suffix=suffix,)

if __name__ == "__main__":
    # Point d'entrée : lance la validation croisée imbriquée pour chaque sujet
    # et exporte les cartes cérébrales (R², alphas, TSNR) ainsi que les histogrammes.

    # --- PARAMÈTRES ---
    plateforme = ["Rorqual", "Mac"]
    plateforme = plateforme[0]

    liste_sujets = ["sub-01", "sub-02", "sub-03", "sub-06"]
    liste_sujets = liste_sujets[2:3]
    LAYER = "encoder_layer7_ffn"

    flag_delai_bold_brute = True
    centrage_donne_temps  = False
    flag_precision_voxel  = False
    randomize_flag        = False
    ROImask_flag          = False

    liste_ROI = ["faceFFA", "scenePPA", "bodyEBA", "V1", "V2", "V3",
                 "hv4", "dorsalAttention", "ventralAttention", "visual"]

    alphas_par_sujet_voxel = {
        "sub-01": np.logspace(2, 9, 20),
        "sub-02": np.logspace(1, 8, 20),
        "sub-03": np.logspace(0, 7, 20),
        "sub-06": np.logspace(2, 9, 20),
    }
    alphas_par_sujet_parcelle = {
        "sub-01": np.logspace(2, 7, 20),
        "sub-02": np.logspace(1, 6, 20),
        "sub-03": np.logspace(1, 4, 20),
        "sub-06": np.logspace(2, 5, 20),
    }

    for SUB in liste_sujets:
        print(f"\n{'='*60}\n  Sujet : {SUB}\n{'='*60}")

        #alphas = alphas_par_sujet_voxel[SUB] if flag_precision_voxel else alphas_par_sujet_parcelle[SUB]
        alphas = np.logspace(-1, 10, 20)
        # ── Alignement normal ────────────────────────────────────────────────
        ridge = RidgeRegression(
            plateforme, SUB, LAYER,
            flag_delai_bold_brute, centrage_donne_temps,
            flag_precision_voxel, ROImask_flag, randomize_flag=False
        )

        print("\n[TEST] nested_cross_validation")
        r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, tsnr = ridge.nested_cross_validation(alphas, 10, 0.1)

        # Moyenne géométrique sur les folds (les alphas s'étalent sur plusieurs décades)
        alphas_moyens = 10 ** np.mean(np.log10(alphas_tous_externes), axis=0)

        print(f"Variance inter-folds moyenne : {np.mean(r2_variance_inter_folds):.6f}")

        # ── Génération de l'ensemble des figures ──────────────────────────
        print("\n[GÉNÉRATION DES FIGURES]")

        # 1. Cartes Cérébrales (Nilearn 3D)
        print(" -> Création des cartes cérébrales (R², Alphas, TSNR)...")
        ridge.brain_mapping_r2(r2_moyen, suffix="_nested_moyen")
        ridge.brain_mapping_alphas(alphas_moyens, suffix="_nested_moyen")
        ridge.brain_mapping_tsnr(tsnr, suffix="_nested")

        # 2. Histogrammes des paramètres de régularisation (Alphas)
        print(" -> Création des histogrammes des alphas...")
        ridge.plot_alphas_histogram(alphas_fold=alphas_tous_externes, grille_alphas=alphas, suffix="_nested_folds")
        ridge.plot_alphas_histogram(alphas_fold=None, grille_alphas=alphas, alphas_finaux=alphas_moyens, suffix="_nested_moyen")

        # 3. Métriques de performance R²
        print(" -> Création des graphiques de distribution de l'accuracy...")
        ridge.plot_r2_distribution(r2_tous_les_tests, suffix="_nested")
        ridge.plot_r2_threshold(r2_tous_les_tests, suffix="_nested")
        ridge.plot_accuracy(r2_tous_les_tests)

        # 4. Analyse par Région d'Intérêt (ROI)
        print(" -> Création de l'analyse par ROI...")
        if flag_precision_voxel:
            # Cette fonction nécessite les données au niveau du voxel
            ridge.plot_ROImask_histogram(r2_moyen)
        else:
            print(" -> (Ignoré : l'analyse par ROI nécessite flag_precision_voxel = True)")

        print(f"\nTerminé pour le sujet {SUB}. Toutes les figures ont été sauvegardées.")
        """
        print("\n[ÉTAPE 1] Cross-validation — optimisation des alphas")
        scores_r2, r2_fold, alphas_finaux, alphas_fold = ridge.cross_validation(alphas)
        ridge.plot_alphas_histogram(alphas_fold, grille_alphas=alphas, suffix="_cv")

        print("\n[ÉTAPE 2] Évaluation finale stricte sur sessions 14-15-16")
        r2_test, alphas_utilises = ridge.evaluation_finale(alphas_finaux)

        ridge.brain_mapping_r2(r2_test, suffix="_test_final")
        ridge.brain_mapping_alphas(alphas_utilises, suffix="_test_final")
        ridge.plot_alphas_histogram(alphas_fold=None, grille_alphas=alphas,
                                    alphas_finaux=alphas_utilises, suffix="_test_final")
        ridge.plot_ROImask_histogram(r2_test, liste_ROI)

        print("\n[ÉTAPE 3] TSNR")
        ridge.brain_mapping_tsnr()

        # ── Alignement randomisé (baseline) ─────────────────────────────────
        print("\n[ÉTAPE 4] Baseline — alignement randomisé")
        ridge_random = RidgeRegression(
            plateforme, SUB, LAYER,
            flag_delai_bold_brute, centrage_donne_temps,
            flag_precision_voxel, ROImask_flag, randomize_flag=True
        )

        scores_r2_random, _, alphas_finaux_random, alphas_fold_random = ridge_random.cross_validation(alphas)
        r2_test_random, _ = ridge_random.evaluation_finale(alphas_finaux_random)

        ridge_random.brain_mapping_r2(r2_test_random, suffix="_randomise")
        """
