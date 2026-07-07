"""
Régression Ridge pour l'encodage cérébral THINGS memory.
Entraîne une RidgeCV par couche et évalue la prédiction.
"""
from pathlib import Path
from TribeHDF5Normalization import TribeHDF5Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import h5py
import gc
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.plotting import plot_stat_map
import matplotlib
from dataclasses import dataclass

matplotlib.use('Agg')

@dataclass
class CheminsProjet:
    root_encoding: Path
    root_timeseries: Path
    chemin_tribe: Path
    chemin_cneuromod: Path
    chemin_atlas: Path

class RidgeRegression:

    def __init__(self, plateforme, subject, layer,  flag_delai_bold_brute, centrage_donne_temps, flag_precision_voxel):
        self.plateforme = plateforme
        self.subject = subject
        self.layer = layer
        self.flag_delai_bold_brute = flag_delai_bold_brute
        self.centrage_donne_temps = centrage_donne_temps
        self.flag_precision_voxel = flag_precision_voxel


    def get_path_file_by_plateform(self, plateforme):
        if plateforme == "Roquale":
            ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
            ROOT_TIMESERIES = Path("/home/aclaud/links/scratch/things.timeseries")
        else:
            ROOT_ENCODING = Path(__file__).parent.parent
            ROOT_TIMESERIES = ROOT_ENCODING

        chemin_tribe = ROOT_ENCODING / "output" / "hdf5" / "things_encoding" / f"{self.subject}.h5"

        if self.flag_precision_voxel:
            sous_dossier = ROOT_TIMESERIES / "timeseries" / "voxel_native" / self.subject
            chemin_cneuromod = sous_dossier / f"{self.subject}_task-things_space-T1w_desc-voxelwise_timeseries.h5"
            chemin_atlas = sous_dossier / f"{self.subject}_task-things_space-T1w_label-GMfromFS_desc-indivFunc_mask.nii.gz"
        else:
            sous_dossier = ROOT_TIMESERIES / "timeseries" / "cneuromod2026" / self.subject
            chemin_cneuromod = sous_dossier / f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
            chemin_atlas = sous_dossier / f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_dseg.nii.gz"

        return CheminsProjet(ROOT_ENCODING, ROOT_TIMESERIES, chemin_tribe, chemin_cneuromod, chemin_atlas)

    def get_chemin_annotations_parcelles(self, plateforme):
        nom_fichier_annotations = (
            "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018TianS3NettekovenAsym_"
            "desc-1000Parcels7Networks50Subcort128Cereb_parcelAnnotations.tsv"
        )
        if plateforme == "Roquale":
            ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
            return ROOT_ENCODING / "data" / "brain_map_subj" / nom_fichier_annotations
        else:
            ROOT = Path(__file__).parent.parent
            return ROOT / "data" / "brain_map_subj" / nom_fichier_annotations

    def charger_noms_parcelles(self, plateforme):
        chemin_annotations = self.get_chemin_annotations_parcelles(plateforme)
        annotations = pd.read_csv(chemin_annotations, sep="\t")
        return annotations["name"].tolist()

    def discover_runs(self,tribe_hdf5=None):
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
                    if self.plateforme == "Roquale":
                        chemin_video = chemins.root_encoding / "data" / "data" / self.subject / tribe_ses / nom_video
                    else:
                        chemin_video = chemins.root_encoding / "data" / self.subject / tribe_ses / nom_video

                    runs.append((tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset))
        finally:
            if gere_localement:
                tribe_hdf5.close()

        print(f"{len(runs)} runs trouvés dans {chemins.chemin_tribe.name}")
        return runs

    def prepare_X_and_Y(self):

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
                if not chemin_video.exists():
                    print(f"Vidéo manquante, run ignoré : {chemin_video.name}")
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

            X = np.concatenate(X_list, axis=0)
            Y = np.concatenate(Y_list, axis=0)
            groupes = np.concatenate(groupes_list, axis=0)
            print(f"Matrice finale : X={X.shape}, Y={Y.shape}")

            del X_list, Y_list, groupes_list

            return runs_ok, X, Y, groupes

    def ridge_regression(self, PCA_flag, alphas, X_train, Y_train, X_test, Y_test,taille_lot=5000):
        # 2. Standardisation
        scaler_X = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        X_test_scaled = scaler_X.transform(X_test)

        if PCA_flag:
            pca = PCA(n_components=0.95)  # garde 95% de la variance
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
            print(f"      [PCA] Dimensions réduites de {X_train.shape[1]} à {X_train_scaled.shape[1]}")

        n_cibles = Y_train.shape[1]
        scores_r2 = np.zeros(n_cibles, dtype=np.float32)

        # Accumulateurs pour le diagnostic global des alphas (sur tous les lots)
        alphas_tous_lots = np.zeros(n_cibles, dtype=np.float64)

        n_lots = int(np.ceil(n_cibles / taille_lot))

        for i, debut in enumerate(range(0, n_cibles, taille_lot)):
            fin = min(debut + taille_lot, n_cibles)

            modele = RidgeCV(alphas=alphas, alpha_per_target=True)
            modele.fit(X_train_scaled, Y_train[:, debut:fin])

            alphas_tous_lots[debut:fin] = modele.alpha_

            Y_pred_lot = modele.predict(X_test_scaled)
            scores_r2[debut:fin] = r2_score(
                Y_test[:, debut:fin], Y_pred_lot, multioutput="raw_values"
            )

            del modele, Y_pred_lot
            gc.collect()

            if i % 5 == 0 or i == n_lots - 1:
                print(f"      [Ridge] Lot {i + 1}/{n_lots} traité (cibles {debut}-{fin})")

        # --- Diagnostic : où se situent les alphas optimaux ? ---
        print(f"[Diagnostic alphas] min={alphas_tous_lots.min():.2e}, "
              f"max={alphas_tous_lots.max():.2e}, "
              f"médiane={np.median(alphas_tous_lots):.2e}")

        # Combien de cibles ont choisi les bornes extrêmes de ta grille ?
        n_borne_min = np.sum(alphas_tous_lots == alphas.min())
        n_borne_max = np.sum(alphas_tous_lots == alphas.max())
        print(f"[Diagnostic alphas] {n_borne_min}/{len(alphas_tous_lots)} cibles à la borne MIN "
              f"({alphas.min():.2e}), {n_borne_max}/{len(alphas_tous_lots)} à la borne MAX "
              f"({alphas.max():.2e})")
        # ----------------------------------------------------------

        del X_train_scaled, X_test_scaled
        gc.collect()

        return scores_r2


    def cross_validation(self, mode, cv_type, alphas, PCA_flag):
        if mode == "train":
            if cv_type == "LeaveOneGroupOut":
                logo = LeaveOneGroupOut()
                runs_ok, X, Y, groupes = self.prepare_X_and_Y()

                if len(X) == 0 or len(Y) == 0:
                    print("Erreur X ou X ne contient rien ")
                    return None

                n_folds = len(np.unique(groupes))
                print(f"\n[Validation Croisée] Lancement sur {n_folds} sessions (Test sur 1 session par fold)...")

                scores_tous_les_folds = []

                for index_fold, (train_index, test_index) in enumerate(logo.split(X, Y, groupes)):
                    print(f"\n--- Évaluation du Fold {index_fold + 1}/{n_folds} ---")

                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    scores_fold = self.ridge_regression(PCA_flag, alphas, X_train, Y_train, X_test, Y_test)
                    scores_tous_les_folds.append(scores_fold)
                    print(f"    -> R² max sur ce fold : {np.max(scores_fold):.5f}")

                scores_finaux = np.mean(scores_tous_les_folds, axis=0)

                del X, Y, groupes

                return scores_finaux

            elif cv_type == "CustomHoldout":
                runs_ok, X, Y, groupes = self.prepare_X_and_Y()

                if len(X) == 0 or len(Y) == 0:
                    print("Erreur X ou X ne contient rien ")
                    return None

                train_sessions = list(range(1, 13)) + list(range(18, 37))
                test_sessions = [14, 15, 16]

                train_mask = np.isin(groupes, train_sessions)
                test_mask = np.isin(groupes, test_sessions)

                X_train, Y_train = X[train_mask], Y[train_mask]
                X_test, Y_test = X[test_mask], Y[test_mask]

                print(f"\n[Évaluation Custom] Mémorisation (Train) : Sessions {train_sessions}")
                print(f"                    Prédiction (Test)  : Sessions {test_sessions}")
                print(f"                    Ignorées (Gaps)    : Sessions 13 et 17")
                print(f"--- Dimensions | Train : {X_train.shape[0]} TRs | Test : {X_test.shape[0]} TRs ---")

                scores_finaux = self.ridge_regression(PCA_flag, alphas, X_train, Y_train, X_test, Y_test)

                del X, Y, groupes

                return scores_finaux
        return None



    def print_scores(self, scores_finaux, noms_parcelles=None):
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

    def brain_mapping(self, scores_r2):
        if self.flag_precision_voxel == False:
            noms_parcelles = self.charger_noms_parcelles(self.plateforme)
            self.print_scores(scores_r2, noms_parcelles)

            _, _, _, _, atlas_path = self.get_path_file_by_plateform(self.plateforme)

            atlas_masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=False)
            atlas_masker.fit()

            r2_map_3d = atlas_masker.inverse_transform(scores_r2)

            display = plot_stat_map(
                r2_map_3d,
                threshold=0.01,
                vmin=0,
                vmax=np.max(scores_r2),
                symmetric_cbar=False,
                display_mode='mosaic',
                title=f'R² Map pour {self.subject} - {self.layer}',
                colorbar=True,
                cmap='YlOrRd',
            )
            display.savefig(f"../output/brain_map_{self.subject}_{self.layer}_parcelles.png", dpi=300)
            display.close()
            print(f"Carte cérébrale sauvegardée : brain_map_{self.subject}_{self.layer}_parcelles.png")

        else:
            self.print_scores(scores_r2, noms_parcelles=None)

            _, _, _, _, masque_path = self.get_path_file_by_plateform(self.plateforme)

            masker = NiftiMasker(mask_img=masque_path, standardize=False)
            masker.fit()

            r2_map_3d = masker.inverse_transform(scores_r2)

            display = plot_stat_map(
                r2_map_3d,
                threshold=0.01,
                vmin=0,
                vmax=np.max(scores_r2),
                symmetric_cbar=False,
                display_mode='mosaic',
                title=f'R² Map pour {self.subject} - {self.layer}',
                colorbar=True,
                cmap='YlOrRd',
            )
            display.savefig(f"../output/brain_map_{self.subject}_{self.layer}_voxel.png", dpi=300)
            display.close()
            print(f"Carte cérébrale sauvegardée : brain_map_{self.subject}_{self.layer}_voxel.png")

if __name__ == "__main__":

    # --- PARAMÈTRES ML ---
    alphas = np.logspace(-1, 20, 20)

    # Chemins
    plateforme = ["Roquale", "Mac"]
    plateforme = plateforme[1]

    SUB = "sub-03"
    LAYER = "encoder_layer7_ffn"

    flag_delai_bold_brute = True
    centrage_donne_temps = False
    flag_precision_voxel = False

    mode = "train"
    cv_type = "CustomHoldout"
    PCA_flag = False

    ridge = RidgeRegression(plateforme, SUB, LAYER, flag_delai_bold_brute, centrage_donne_temps, flag_precision_voxel)

    scores_r2 = ridge.cross_validation(mode, cv_type, alphas, PCA_flag)

    if scores_r2 is not None:
        ridge.brain_mapping(scores_r2)
