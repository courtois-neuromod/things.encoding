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
import h5py
import gc
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.decomposition import PCA
from nilearn.maskers import NiftiLabelsMasker
from nilearn.plotting import plot_stat_map, show
import matplotlib
matplotlib.use('Agg')

class RidgeRegression:

    def __init__(self, plateforme, subject, layer,  flag_delai_bold_brute, centrage_donne_temps):
        self.plateforme = plateforme
        self.subject = subject
        self.layer = layer
        self.flag_delai_bold_brute = flag_delai_bold_brute
        self.centrage_donne_temps = centrage_donne_temps

    def get_path_file_by_plateform(self, plateforme):
        if plateforme == "Roquale":
            ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
            ROOT_TIMESERIES = Path("/home/aclaud/links/scratch/things.timeseries")

            chemin_tribe = ROOT_ENCODING / "output" / "hdf5" / "things_encoding" / f"{self.subject}.h5"

            chemin_cneuromod = (
                    ROOT_TIMESERIES / "timeseries" / "cneuromod2026" / self.subject /
                    f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
            )

            chemin_atlas = (
                    ROOT_TIMESERIES / "timeseries" / "cneuromod2026" / self.subject /
                    f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_dseg.nii.gz"
            )
            return ROOT_ENCODING, ROOT_TIMESERIES, chemin_tribe, chemin_cneuromod, chemin_atlas
        else:
            ROOT = Path(__file__).parent.parent

            chemin_tribe = ROOT / "output" / "hdf5" / f"{self.subject}.h5"

            chemin_cneuromod = (
                    ROOT / "data" / "timeseries" / "cneuromod2026" / self.subject /
                    f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"
            )

            chemin_atlas = (
                    ROOT / "data" / "timeseries" / "cneuromod2026" / self.subject /
                    f"{self.subject}_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_dseg.nii.gz"
            )
            return ROOT, chemin_tribe, chemin_cneuromod, chemin_atlas

    def discover_runs(self):
        if self.plateforme == "Roquale":
            ROOT_ENCODING, ROOT_TIMESERIES, chemin_tribe, chemin_cneuromod, _ = self.get_path_file_by_plateform(self.plateforme)
        else :
            ROOT, chemin_tribe, chemin_cneuromod, _ = self.get_path_file_by_plateform(self.plateforme)

        if not chemin_tribe.exists():
            print(f"Erreur : Fichier introuvable : {chemin_tribe}")

        runs = []

        with h5py.File(chemin_tribe, "r") as f:
            for tribe_ses in sorted(f.keys()):           # "ses-001", "ses-002", ...
                for tribe_run in sorted(f[tribe_ses].keys()):   # "run-1", "run-2", ...

                    # Conversion ses-001 → ses-01 pour CNeuroMod
                    num_ses = int(tribe_ses.replace("ses-", ""))
                    cneuromod_ses = f"ses-{num_ses:02d}"

                    # Clé dataset CNeuroMod
                    num_run = tribe_run.replace("run-", "")
                    cneuromod_dataset = f"{cneuromod_ses}_task-things_run-{num_run}_timeseries"

                    # Chemin vidéo originale (non CFR) pour ffprobe
                    nom_video = f"{self.subject}_{tribe_ses}_task-thingsmemory_{tribe_run}.mp4"
                    if self.plateforme == "Roquale":
                        chemin_video = ROOT_ENCODING / "data" / "data" / self.subject / tribe_ses / nom_video
                    else :
                        chemin_video = ROOT / "data" / self.subject / tribe_ses / nom_video

                    runs.append((tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset))

            print(f"{len(runs)} runs trouvés dans {chemin_tribe.name}")
            return runs

    def prepare_X_and_Y(self):

        if self.plateforme == "Roquale":
            ROOT_ENCODING, ROOT_TIMESERIES, chemin_tribe, chemin_cneuromod, _ = self.get_path_file_by_plateform(self.plateforme)
        else :
            ROOT, chemin_tribe, chemin_cneuromod, _ = self.get_path_file_by_plateform(self.plateforme)

        runs = self.discover_runs()

        # Alignement temporel et concaténation
        X_list, Y_list = [], []
        runs_ok = []
        groupes_list = []

        for id_run, (tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset) in enumerate(runs):
            # Vérifier que la vidéo source existe localement
            if not chemin_video.exists():
                print(f"Vidéo manquante, run ignoré : {chemin_video.name}")
                continue

            normalisateur = TribeHDF5Normalization(
                chemin_tribe=chemin_tribe,
                chemin_cneuromod=chemin_cneuromod,
                chemin_video=chemin_video,
                tribe_ses=tribe_ses,
                tribe_run=tribe_run,
                tribe_layer=self.layer,
                cneuromod_ses=cneuromod_ses,
                cneuromod_dataset=cneuromod_dataset,
                t_Tribe_s=0.5,
                TR_irmf_s=1.49,
                flag_delai_bold_brute=True,
                centrage_donne_temps=False,
            )
            X_run, Y_run = normalisateur.executer_pipeline()
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
        gc.collect()

        return runs_ok, X, Y, groupes

    def ridge_regression(self, PCA_flag, alphas, X_train, Y_train, X_test, Y_test):
        # 2. Standardisation
        scaler_X = StandardScaler()
        #scaler_Y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        #Y_train_scaled = scaler_Y.fit_transform(Y_train)
        X_test_scaled = scaler_X.transform(X_test)
        #Y_test_scaled = scaler_Y.transform(Y_test)

        if PCA_flag:
            pca = PCA(n_components=0.95)  # garde 95% de la variance
            X_train_reduit = pca.fit_transform(X_train_scaled)
            X_test_reduit = pca.transform(X_test_scaled)

            if X_train_reduit.shape[1] < X_train.shape[1]:
                print(f"      [PCA] Dimensions réduites de {X_train.shape[1]} à {X_train_reduit.shape[1]}")

            modele = RidgeCV(alphas=alphas, alpha_per_target=True)
            modele.fit(X_train_reduit, Y_train_scaled)

            Y_pred_scaled = modele.predict(X_test_reduit)
        else:
            modele = RidgeCV(alphas=alphas, alpha_per_target=True)
            #modele.fit(X_train_scaled, Y_train_scaled)
            modele.fit(X_train_scaled, Y_train)
            Y_pred_scaled = modele.predict(X_test_scaled)

        #scores_r2 = r2_score(Y_test_scaled, Y_pred_scaled, multioutput="raw_values")
        scores_r2 = r2_score(Y_test, Y_pred_scaled, multioutput="raw_values")
        #del X_train_scaled, Y_train_scaled, X_test_scaled, Y_test_scaled, modele
        del X_train_scaled, Y_pred_scaled, X_test_scaled
        gc.collect()

        return scores_r2


    def cross_validation(self, mode, type, alphas, PCA_flag):
        if mode == "train":
            if type == "LeaveOneGroupOut":
                logo = LeaveOneGroupOut()
                runs_ok, X, Y, groupes = self.prepare_X_and_Y()

                if len(X) == 0 or len(Y) == 0:
                    print("Erreur X ou X ne contient rien ")
                    return None

                print(f"\n[Validation Croisée] Lancement sur {len(np.unique(groupes))} sessions (Test sur 1 session par fold)...")

                scores_tous_les_folds = []
                for index_fold, (train_index, test_index) in enumerate(logo.split(X, Y, groupes)):
                    print(f"\n--- Évaluation du Fold {index_fold + 1}/{len(runs_ok)} ---")

                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    scores_fold = self.ridge_regression(PCA_flag, alphas, X_train, Y_train, X_test, Y_test)
                    scores_tous_les_folds.append(scores_fold)
                    print(f"    -> R² max sur ce fold : {np.max(scores_fold):.5f}")

                scores_finaux = np.mean(scores_tous_les_folds, axis=0)

                del X, Y, groupes
                gc.collect()

                return scores_finaux

            elif type == "CustomHoldout":
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
                gc.collect()

                return scores_finaux
        return None



    def print_scores(self, mode, type, alphas, PCA_flag):
        scores_finaux =  self.cross_validation(mode, type, alphas, PCA_flag)
        score_moyen = np.mean(scores_finaux)
        score_median = np.median(scores_finaux)
        score_max = np.max(scores_finaux)
        parcelle_max = np.argmax(scores_finaux)
        n_positifs = np.sum(scores_finaux > 0)

        print(f"\n=========================================")
        print(f"[Résultats Finaux Robustes — couche {self.layer}]")
        print(f"R² moyen   : {score_moyen:.4f}")
        print(f"R² médian  : {score_median:.4f}")
        print(f"R² max     : {score_max:.4f}  (parcelle {parcelle_max})")
        print(f"Parcelles R² > 0 : {n_positifs} / {len(scores_finaux)}")
        print(f"=========================================")

if __name__ == "__main__":

    # --- PARAMÈTRES ML ---
    alphas = np.logspace(-1, 20, 20)
    logo = LeaveOneGroupOut()

    # Chemins
    plateforme = ["Roquale", "Mac"]
    plateforme = plateforme[0]

    SUB = "sub-03"
    LAYER = "encoder_layer7_ffn"

    flag_delai_bold_brute = True
    centrage_donne_temps = False

    mode = "train"
    type = "LeaveOneGroupOut"
    PCA_flag = False

    ridge = RidgeRegression(plateforme, SUB, LAYER, flag_delai_bold_brute, centrage_donne_temps)
    ridge.print_scores(mode, type, alphas, PCA_flag)
    
    scores_r2 = ridge.cross_validation(mode, type, alphas, PCA_flag)

    if scores_r2 is not None:
        # Chemin vers ton atlas correspondant (le même que dans ton pipeline)
        _,_,_,_,atlas_path = ridge.get_path_file_by_plateform(plateforme)

        # Création du masker
        atlas_masker = NiftiLabelsMasker(labels_img=atlas_path, standardize=False)
        atlas_masker.fit()

        # Projection 1D -> 3D
        r2_map_3d = atlas_masker.inverse_transform(scores_r2)

        # Plot
        display = plot_stat_map(
            r2_map_3d,
            threshold=0.01,  # Filtre les résultats non significatifs
            display_mode='mosaic',
            title=f'R² Map pour {SUB} - {LAYER}',
            colorbar=True,
            cmap='coolwarm',
        )
        display.savefig(f"../output/brain_map_{SUB}_{LAYER}_2.png", dpi=300)
        display.close()
        print(f"Carte cérébrale sauvegardée : brain_map_{SUB}_{LAYER}.png")
