"""
Régression Ridge pour l'encodage cérébral THINGS memory.
Entraîne une RidgeCV par couche et évalue la prédiction.
"""
from pathlib import Path
from TribeHDF5Normalization import TribeHDF5Normalization
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import LeaveOneGroupOut, cross_validate
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

    def create_X_Y_total(self, ):

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

            groupes = np.concatenate(groupes_list, axis=0)
            print(f"Matrice finale : X={X.shape}, Y={Y.shape}")

            del X_list, Y_list, groupes_list

            return runs_ok, X, Y, groupes

    def _selection_X_Y(self, sessions_a_exclure):
        runs_ok, X, Y, groupes = self.create_X_Y_total()
        masque = ~np.isin(groupes, sessions_a_exclure)
        X, Y, groupes = X[masque], Y[masque], groupes[masque]
        return X, Y, groupes

    def _scaler_X_Y(self, X, Y, train_mask, test_mask):
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

    def cross_validation(self, alphas, taille_lot = 5000):

        # Masque : on exclut les sessions réservées au test final et les gaps
        sessions_a_exclure = [13, 14, 15, 16, 17]
        X, Y, groupes = self._selection_X_Y(sessions_a_exclure)

        sessions = np.unique(groupes)
        n_features = Y.shape[1]
        n_folds = len(sessions)

        r2_fold = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_fold = np.zeros((n_folds, n_features), dtype=np.float64)

        for index_fold, session_test in enumerate(sessions):

            train_mask = groupes != session_test
            test_mask = groupes == session_test

            X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test = self._scaler_X_Y(X, Y, train_mask, test_mask)

            r2_fold[index_fold], alphas_fold[index_fold] = self._ridge_par_lots(
                X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test,
                alphas=alphas, taille_lot=taille_lot,
                n_folds=n_folds, index_fold=index_fold,
            )

            print(f"  Fold {index_fold} Session {session_test:02d} en test → R² max : {np.max(r2_fold[index_fold]):.5f}")

        scores_r2_finaux = np.mean(r2_fold, axis=0)
        alphas_finaux = 10 ** np.mean(np.log10(alphas_fold), axis=0)

        return scores_r2_finaux, r2_fold, alphas_finaux, alphas_fold

    def evaluation_finale(self, alphas_optimaux, taille_lot = 5000):
        # Masque : on exclut les sessions réservées au test final et les gaps
        sessions_exclues = [13, 17]
        sessions_evaluation = [14, 15, 16]

        X, Y, groupes = self._selection_X_Y(sessions_exclues)

        train_masque = ~np.isin(groupes, sessions_evaluation)
        test_masque = np.isin(groupes, sessions_evaluation)

        X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test = self._scaler_X_Y(X, Y, train_masque, test_masque)

        print(f"Train : {X_scaled_train.shape[0]} TRs | Test : {X_scaled_test.shape[0]} TRs")

        r2_final, alphas_utilises = self._ridge_par_lots(
            X_scaled_train, X_scaled_test, Y_scaled_train, Y_scaled_test,
            alphas=alphas_optimaux, taille_lot=taille_lot,
        )

        return r2_final, alphas_utilises


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

                fig = px.box(df, x="ROI", y="r2", color="ROI", points="all")
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
    plateforme = "Rorqual"
    liste_sujets = ["sub-03"]
    LAYER = "encoder_layer7_ffn"

    flag_delai_bold_brute = True
    centrage_donne_temps = False
    flag_precision_voxel = False
    randomize_flag = False
    ROImask_flag = True

    liste_ROI = ["faceFFA", "scenePPA", "bodyEBA", "V1", "V2", "V3", "hv4", "dorsalAttention", "ventralAttention", "visual"]

    if flag_precision_voxel == True:
        alphas_par_sujet = {
            "sub-01": np.logspace(2, 9, 20),
            "sub-02": np.logspace(1, 8, 20),
            "sub-03": np.logspace(0, 7, 20),
            "sub-06": np.logspace(2, 9, 20),
        }
    else:
        alphas_par_sujet = {
            "sub-01": np.logspace(2, 7, 20),
            "sub-02": np.logspace(1, 6, 20),
            "sub-03": np.logspace(1, 4, 20),
            "sub-06": np.logspace(2, 5, 20),
        }

    # --- BOUCLE SUJETS ---
    for SUB in liste_sujets:
        print(f"\n{'=' * 60}\n  Sujet : {SUB}\n{'=' * 60}")

        #alphas = alphas_par_sujet[SUB]
        alphas = np.logspace(1, 20, 20)
        ridge = RidgeRegression(
            plateforme, SUB, LAYER,
            flag_delai_bold_brute, centrage_donne_temps,
            flag_precision_voxel, ROImask_flag, randomize_flag
        )

        # Étape 1 — boucle interne : trouve les alphas optimaux
        print("\n[ÉTAPE 1] Cross-validation interne — optimisation des alphas")
        scores_r2, r2_fold, alphas_finaux, alphas_fold = ridge.cross_validation(alphas)

        #ridge.brain_mapping_r2(scores_r2)
        #ridge.plot_alphas_histogram(alphas_fold, grille_alphas=alphas)

        # Étape 2 — entraînement final + test strict sur sessions 14-15-16
        print("\n[ÉTAPE 2] Évaluation finale stricte sur sessions 14-15-16")
        r2_test, alphas_utilises = ridge.evaluation_finale(alphas_finaux)

        ridge.brain_mapping_r2(r2_test, suffix="_test_final")
        ridge.brain_mapping_alphas(alphas_utilises, suffix="_test_final")
        ridge.plot_alphas_histogram([alphas_utilises], grille_alphas=alphas, suffix="_test_final")
        ridge.plot_ROImask_histogram(r2_test, liste_ROI)
