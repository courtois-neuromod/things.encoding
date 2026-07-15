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
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from dataclasses import dataclass
import plotly.express as px

matplotlib.use('Agg')

@dataclass
class CheminsProjet:
    root_encoding: Path
    root_timeseries: Path
    chemin_tribe: Path
    chemin_cneuromod: Path
    chemin_atlas: Path
    chemin_ROImask: Path
    chemin_anatomie: Path = None


class RidgeRegression:

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

    def prepare_X_and_Y(self,):

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
                print(f"⚠Baseline activée : Y_list réordonné aléatoirement ({nombre_de_runs} runs, aucun n'a gardé sa position d'origine)")

            X = np.concatenate(X_list, axis=0)
            Y = np.concatenate(Y_list, axis=0)

            groupes = np.concatenate(groupes_list, axis=0)
            print(f"Matrice finale : X={X.shape}, Y={Y.shape}")

            del X_list, Y_list, groupes_list

            return runs_ok, X, Y, groupes

    def ridge_regression(self, PCA_flag, alphas, X_train, Y_train, X_test, Y_test,taille_lot=5000):
        # 2. Standardisation
        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        X_train_scaled = scaler_X.fit_transform(X_train)
        Y_train_scaled = scaler_Y.fit_transform(Y_train)
        #Y_train_scaled = Y_train
        X_test_scaled = scaler_X.transform(X_test)
        Y_test_scaled = scaler_Y.transform(Y_test)
        #Y_test_scaled = Y_test

        if PCA_flag:
            pca = PCA(n_components=0.95)  # garde 95% de la variance
            X_train_scaled = pca.fit_transform(X_train_scaled)
            X_test_scaled = pca.transform(X_test_scaled)
            print(f"      [PCA] Dimensions réduites de {X_train.shape[1]} à {X_train_scaled.shape[1]}")

        n_cibles = Y_train_scaled.shape[1]
        scores_r2 = np.zeros(n_cibles, dtype=np.float32)

        # Accumulateurs pour le diagnostic global des alphas (sur tous les lots)
        alphas_tous_lots = np.zeros(n_cibles, dtype=np.float64)

        n_lots = int(np.ceil(n_cibles / taille_lot))

        for i, debut in enumerate(range(0, n_cibles, taille_lot)):
            fin = min(debut + taille_lot, n_cibles)

            modele = RidgeCV(alphas=alphas, alpha_per_target=True)
            modele.fit(X_train_scaled, Y_train_scaled[:, debut:fin])

            alphas_tous_lots[debut:fin] = modele.alpha_

            Y_pred_lot = modele.predict(X_test_scaled)
            scores_r2[debut:fin] = r2_score(
                Y_test_scaled[:, debut:fin], Y_pred_lot, multioutput="raw_values"
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

        return scores_r2, alphas_tous_lots


    def cross_validation(self, mode, cv_type, alphas, PCA_flag):
        if mode == "train":
            if cv_type == "LeaveOneGroupOut":
                logo = LeaveOneGroupOut()
                runs_ok, X, Y, groupes = self.prepare_X_and_Y()

                if len(X) == 0 or len(Y) == 0:
                    print("Erreur X ou X ne contient rien ")
                    return None, None

                n_folds = len(np.unique(groupes))
                print(f"\n[Validation Croisée] Lancement sur {n_folds} sessions (Test sur 1 session par fold)...")

                scores_tous_les_folds = []
                alphas_tous_les_folds = []
                for index_fold, (train_index, test_index) in enumerate(logo.split(X, Y, groupes)):
                    print(f"\n--- Évaluation du Fold {index_fold + 1}/{n_folds} ---")

                    X_train, X_test = X[train_index], X[test_index]
                    Y_train, Y_test = Y[train_index], Y[test_index]

                    scores_fold, alphas_fold = self.ridge_regression(PCA_flag, alphas, X_train, Y_train, X_test, Y_test)
                    scores_tous_les_folds.append(scores_fold)
                    alphas_tous_les_folds.append(alphas_fold)
                    print(f"    -> R² max sur ce fold : {np.max(scores_fold):.5f}")

                scores_finaux = np.mean(scores_tous_les_folds, axis=0)
                log_alphas = np.log10(alphas_tous_les_folds)
                alphas_finaux = 10 ** np.mean(log_alphas, axis=0)

                del X, Y, groupes

                return scores_finaux, scores_tous_les_folds, alphas_finaux, alphas_tous_les_folds

            elif cv_type == "CustomHoldout":
                runs_ok, X, Y, groupes = self.prepare_X_and_Y()

                if len(X) == 0 or len(Y) == 0:
                    print("Erreur X ou X ne contient rien ")
                    return None, None

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

                scores_finaux, alphas_finaux = self.ridge_regression(PCA_flag, alphas, X_train, Y_train, X_test, Y_test)

                del X, Y, groupes

                return scores_finaux, [scores_finaux], alphas_finaux, [alphas_finaux]
        return None, None

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

    def plot_ROImask_histogram(self, scores_finaux, liste_ROI):
        chemins = self.get_path_file_by_plateform(self.plateforme)
        fichier_ROImask = chemins.chemin_ROImask

        ROIs_noms = []
        ROIs_vecteur = []

        if flag_precision_voxel:
            with h5py.File(fichier_ROImask, 'r') as fichier:
                for groupe in fichier.keys():
                    for sous_cle in fichier[groupe].keys():
                        ROI_dataset = fichier[groupe][sous_cle]
                        if sous_cle in liste_ROI:
                            vecteur = ROI_dataset[:].astype(bool)
                            scores_roi = scores_finaux[vecteur]
                            ROIs_noms.extend([sous_cle] * len(scores_roi))
                            ROIs_vecteur.extend(scores_roi)
                df = pd.DataFrame({"ROI": ROIs_noms, "r2": ROIs_vecteur})
                print(df.shape)

                fig = px.box(df, x="ROI", y="r2", color="ROI", points=False)
                fig.write_html(str(chemins.root_encoding / "output" / f"ROImask_{self.subject}.html"))
            print("Histogramme ROI sauvegardé :", str(chemins.root_encoding / "output" / f"ROImask_{self.subject}.html"))
            return
        else:
            return "Pas en voxel"


    def plot_alphas_histogram(self, alphas_par_fold, grille_alphas, suffix=""):
        fig, ax = plt.subplots(figsize=(10, 6), facecolor='white')
        log10_grille = np.log10(grille_alphas)

        step = log10_grille[1] - log10_grille[0]
        bins = np.append(log10_grille - step / 2, log10_grille[-1] + step / 2)

        n_folds = len(alphas_par_fold)
        cmap = plt.get_cmap("tab20" if n_folds <= 20 else "viridis")

        for cv_fold, best_alphas in enumerate(alphas_par_fold):
            log10_best_alphas = np.log10(best_alphas)

            hist, _ = np.histogram(log10_best_alphas, bins=bins)

            couleur = cmap(cv_fold / max(1, n_folds - 1)) if n_folds > 1 else "#d73027"
            label = f"Fold {cv_fold + 1}" if n_folds > 1 else "Test Unique"

            if n_folds == 1:
                # Vrais bâtons pleins classiques pour 1 seul test
                ax.hist(
                    log10_best_alphas, bins=bins, color=couleur, edgecolor="black", alpha=0.8, label=label
                )
            else:
                # Bâtons non-remplis pour superposer plusieurs folds proprement
                ax.hist(
                    log10_best_alphas, bins=bins, histtype="step",color=couleur, linewidth=2, alpha=0.8, label=label
                )

        unite = "voxels" if self.flag_precision_voxel else "parcelles"
        ax.set_ylabel(f"Nombre de {unite}", fontsize=12)
        ax.set_xlabel(r"$\log_{10}(\alpha)$", fontsize=12)

        titre = f"Distribution des Alphas optimaux par Fold\n{self.subject} - {self.layer}"
        ax.set_title(titre, fontsize=15, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.7)

        ax.set_xticks(log10_grille)
        ax.set_xticklabels([f"{x:.1f}" for x in log10_grille], rotation=45)

        if n_folds > 1:
            ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=(n_folds // 15 + 1))
        else:
            ax.legend(fontsize=11)

        nom_fichier = f"histogram_alphas_folds_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sortie = self.get_path_file_by_plateform(self.plateforme).root_encoding / "output" / nom_fichier

        fig.savefig(chemin_sortie, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"Histogramme diagnostic sauvegardé : {chemin_sortie}")

    def _brain_mapping_generique(self, donnees, nom_carte, cmap, treshold = 0.01, echelle_log=False, vmin = None, vmax = None, suffix=""):
        chemins = self.get_path_file_by_plateform(self.plateforme)

        donnees_affichees = np.log10(donnees) if echelle_log else donnees

        if self.flag_precision_voxel == True:
            masker = NiftiMasker(mask_img=chemins.chemin_atlas, standardize=False)
            kwargs_bg = {"bg_img": chemins.chemin_anatomie}
        else:
            masker = NiftiLabelsMasker(labels_img=chemins.chemin_atlas, standardize=False)
            kwargs_bg = {}

        coords_R2_map = {'x': np.array([-52.5, -28.5, -12.5, 9.5, 21.5, 35.5, 47.5]), 'y': np.array([-96.5, -80.5, -60.5, -42.5, -26.5, 53.5, 69.5]), 'z': np.array([-18.5, -4.5, 7.5, 19.5, 31.5, 45.5, 61.5])}
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
            cut_coords = coords_R2_map,
            cbar_tick_format="%.2f",
            colorbar=True,
            cmap=cmap,
            **kwargs_bg,
        )
        unite = "voxel" if self.flag_precision_voxel == True else "parcelle"
        title=f'{nom_carte} pour {self.subject} - {self.layer} en {unite}'
        fig.suptitle(title, fontsize=18, fontweight='bold', color='black', y=0.98, ha='center')
        fig.subplots_adjust(top=0.92)

        if echelle_log and display._cbar is not None:
            display._cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda valeur, position: f"$10^{{{valeur:.0f}}}$"))

        chemin_sortie = f"../output/brain_map_{self.subject}_{self.layer}_{nom_carte}_{unite}{suffix}.png"
        display.savefig(chemin_sortie, dpi=300)
        display.close()
        plt.close(fig)
        print(f"Carte cérébrale sauvegardée : {chemin_sortie}")
        return

    def brain_mapping_r2(self, scores_r2, noms_parcelles=None, suffix=""):
        self.print_scores(scores_r2, noms_parcelles)
        self._brain_mapping_generique(scores_r2, nom_carte="R2", cmap="YlOrRd", treshold=0.01, echelle_log=False, vmin=0, vmax=np.max(scores_r2), suffix=suffix)

    def brain_mapping_alphas(self, alphas_tous_les_lots, suffix=""):
        self._brain_mapping_generique(alphas_tous_les_lots, nom_carte="Alphas", cmap="YlOrRd", treshold=0.01, echelle_log=True, suffix=suffix)



if __name__ == "__main__":

    # --- PARAMÈTRES ---
    plateforme = ["Rorqual", "Mac"]
    plateforme = plateforme[1]

    liste_sujets = ["sub-01", "sub-02", "sub-03", "sub-06"]
    liste_sujets = liste_sujets[2:3]
    LAYER = "encoder_layer7_ffn"

    flag_delai_bold_brute = True
    centrage_donne_temps = False
    flag_precision_voxel = True
    randomize_flag = False
    flag_opt_alphas = False
    ROImask_flag = True

    mode = "train"
    cv_type = "CustomHoldout"
    PCA_flag = False

    liste_ROI = ["faceFFA","scenePPA", "bodyEBA", "V1", "V2", "V3", "hv4", "dorsalAttention", "ventralAttention", "visual" ]

    # Paramètres de la boucle d'optimisation adaptative
    max_iterations = 100
    tolerance_evolution = 1e-4

    for SUB in liste_sujets:
        ridge = RidgeRegression(plateforme, SUB, LAYER, flag_delai_bold_brute, centrage_donne_temps, flag_precision_voxel, ROImask_flag, randomize_flag)

        if flag_opt_alphas:
            # --- MODE OPTIMISATION : boucle adaptive sur la grille d'alphas ---
            start_log, end_log = 5, 6
            meilleur_r2_max = -np.inf
            iteration = 1

            while iteration <= max_iterations:
                alphas = np.logspace(start_log, end_log, 20)
                alpha_min_str = f"{alphas.min():.1e}"
                alpha_max_str = f"{alphas.max():.1e}"
                suffixe_fichier = f"_amin-{alpha_min_str}_amax-{alpha_max_str}"

                print("-" * 113)
                print(f"\n[Sujet: {SUB} | Iteration {iteration}] Test de la plage alphas : [{alpha_min_str} -> {alpha_max_str}]")

                scores_r2, alphas_tous_lots, alphas_tous_les_folds = ridge.cross_validation(mode, cv_type, alphas, PCA_flag)

                if scores_r2 is None:
                    print(f" Echec de l'evaluation pour {SUB}. Passage au sujet suivant.")
                    break

                r2_max_actuel = np.max(scores_r2)
                print(f" -> R2 max mesure a l'iteration {iteration} : {r2_max_actuel:.5f}")
                ridge.brain_mapping_r2(scores_r2, suffix=suffixe_fichier)
                ridge.brain_mapping_alphas(alphas_tous_lots, suffix=suffixe_fichier)
                ridge.plot_alphas_histogram(alphas_tous_les_folds, grille_alphas=alphas, suffix=suffixe_fichier)

                total_cibles = len(alphas_tous_lots)
                n_borne_min = np.sum(alphas_tous_lots == alphas.min())
                n_borne_max = np.sum(alphas_tous_lots == alphas.max())
                seuil_saturation = 0.05 * total_cibles

                gain_r2 = r2_max_actuel - meilleur_r2_max
                if gain_r2 < tolerance_evolution:
                    print(f" Convergence atteinte : gain R2 max = {gain_r2:.5f} < {tolerance_evolution}.")
                    break

                meilleur_r2_max = r2_max_actuel

                if n_borne_max > seuil_saturation and n_borne_min <= seuil_saturation:
                    print(f" -> Saturation haute ({n_borne_max}/{total_cibles} cibles). Decalage vers le haut.")
                    start_log += 1
                    end_log += 1
                elif n_borne_min > seuil_saturation and n_borne_max <= seuil_saturation:
                    print(f" -> Saturation basse ({n_borne_min}/{total_cibles} cibles). Decalage vers le bas.")
                    start_log -= 1
                    end_log -= 1
                elif n_borne_max > seuil_saturation and n_borne_min > seuil_saturation:
                    print(f" -> Double saturation. Elargissement global de la grille.")
                    start_log -= 1
                    end_log += 1
                else:
                    print(" -> Distribution optimale. Fin de la recherche.")
                    break

                iteration += 1

            print(f" Fin de l'optimisation pour {SUB}. Meilleur R2 max : {meilleur_r2_max:.5f}\n")

        else:
            # --- MODE NORMAL : run unique avec la grille fixe ---
            if flag_precision_voxel == False:    
                if SUB == "sub-01":
                    alphas = np.logspace(2, 7, 20)
                elif SUB == "sub-02":
                    alphas = np.logspace(1, 6, 20)
                elif SUB == "sub-03":
                    alphas = np.logspace(1, 4, 20)
                else:
                    alphas = np.logspace(2, 5, 20)
            else:
                if SUB == "sub-01":
                    alphas = np.logspace(2, 9, 20)
                elif SUB == "sub-02":
                    alphas = np.logspace(1, 8, 20)
                elif SUB == "sub-03":
                    alphas = np.logspace(0, 7, 20)
                else:
                    alphas = np.logspace(2, 9, 20)

            scores_r2, alphas_tous_lots, alphas_tous_les_folds = ridge.cross_validation(mode, cv_type, alphas, PCA_flag)

            if scores_r2 is not None:
                ridge.brain_mapping_r2(scores_r2)
                ridge.brain_mapping_alphas(alphas_tous_lots)
                ridge.plot_alphas_histogram(alphas_tous_les_folds, grille_alphas=alphas)
                ridge.plot_ROImask_histogram(scores_r2, liste_ROI)

