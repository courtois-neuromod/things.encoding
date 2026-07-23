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
import seaborn as sns
from sklearn.model_selection import GroupKFold

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

    def nested_cross_validation(self, grille_alphas):
        """Validation croisée imbriquée sur plusieurs seeds.

        Boucle externe : évalue le R² sur une session test jamais vue par le modèle.

        Boucle interne (LeaveOneGroupOut sur les sessions train/val restantes,
        sessions adjacentes au test exclues) : sélectionne l'alpha optimal par
        régularisation par moyenne géométrique des alphas des folds internes.

        Le tout est répété sur plusieurs seeds de tirage des folds externes
        pour estimer la variance des résultats.

        Args:
            grille_alphas: grille de valeurs d'alpha testées par RidgeCV.

        Returns:
            tuple: (r2_moyen, r2_variance_inter_folds, r2_variance_inter_tests,
                r2_tous_les_tests, alphas_tous_externes_moyen, tsnr)
        """
        X, Y, groupes, TSNR = self._selection_X_Y()
        sessions = np.unique(groupes)
        n_sessions = len(sessions)
        n_features = Y.shape[1]

        # Découpage en folds externes (3 sous-groupes)
        n_folds_externes = 3
        #sous_groupes = np.array_split(sessions, n_folds_externes)
        #liste_seed = [42, 16, 28, 32, 12, 70, 56, 69]
        liste_seed = [42, 16]
        n_seed = len(liste_seed)
        r2_tous_les_tests = np.zeros((n_seed, n_folds_externes, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_seed, n_folds_externes, n_features), dtype=np.float64)

        
        for index_seed, seed in enumerate(liste_seed):
            rng = np.random.default_rng(seed)
            sessions_shuffled = rng.permutation(sessions)
            sous_groupes = np.array_split(sessions_shuffled, n_folds_externes)

            liste_numero_test_session = [groupe[0] for groupe in sous_groupes]
            liste_numero_train_val_session = [
                groupe[~np.isin(groupe, liste_numero_test_session)]
                for groupe in sous_groupes
            ]

            print(f"Sessions disponibles : {sessions}")
            print(f"Sessions test par fold externe : {liste_numero_test_session}")

            cv = LeaveOneGroupOut()

            # BOUCLE EXTERNE
            for i_extern, sessions_train_val in enumerate(liste_numero_train_val_session):

                session_test = liste_numero_test_session[i_extern]
                print(f"\n{'=' * 60}")
                print(f"[Fold externe {i_extern + 1}/{n_folds_externes}] Test : {session_test}")

                # Sessions adjacentes au test set → exclues de la boucle interne
                idx_test = np.where(sessions == session_test)[0][0]
                sessions_adjacentes = []
                if idx_test > 0:
                    sessions_adjacentes.append(sessions[idx_test - 1])
                if idx_test < n_sessions - 1:
                    sessions_adjacentes.append(sessions[idx_test + 1])
                sessions_adjacentes = np.array(sessions_adjacentes)

                # Sessions utilisables pour la boucle interne
                sessions_train_val_filtrees = sessions_train_val[
                    ~np.isin(sessions_train_val, sessions_adjacentes)
                ]
                print(f"  Adjacentes exclues : {sessions_adjacentes}")
                print(f"  Train_val boucle interne : {sessions_train_val_filtrees}")

                train_val_masque = np.isin(groupes, sessions_train_val_filtrees)
                X_train_val = X[train_val_masque]
                Y_train_val = Y[train_val_masque]
                groupe_train_val = groupes[train_val_masque]

                # BOUCLE INTERNE : LeaveOneGroupOut
                alphas_tous_folds_int = []

                for index_fold, (train_index, val_index) in enumerate(
                        cv.split(X_train_val, Y_train_val, groupe_train_val)):
                    session_val = np.unique(groupe_train_val[val_index])[0]
                    sessions_train = np.unique(groupe_train_val[train_index])
                    print(f"  [Fold interne {index_fold + 1}] Val : {session_val} | Train : {sessions_train}")

                    train_masque_int = np.zeros(len(X_train_val), dtype=bool)
                    val_masque_int = np.zeros(len(X_train_val), dtype=bool)
                    train_masque_int[train_index] = True
                    val_masque_int[val_index] = True

                    X_sc_train, X_sc_val, Y_sc_train, Y_sc_val = self._scaler_X_Y(
                        X_train_val, Y_train_val,
                        train_masque_int, val_masque_int
                    )

                    modele = RidgeCV(
                        alphas=grille_alphas,
                        alpha_per_target=True,
                        cv=None,
                    )
                    modele.fit(X_sc_train, Y_sc_train)

                    Y_pred = modele.predict(X_sc_val)
                    alphas_int = modele.alpha_
                    alphas_tous_folds_int.append(alphas_int)

                    del modele, Y_pred
                    gc.collect()

                # Alphas optimaux = moyenne géométrique sur folds internes
                alphas_tous_folds_int = np.array(alphas_tous_folds_int)
                alphas_optimaux = 10 ** np.mean(np.log10(alphas_tous_folds_int), axis=0)
                alphas_tous_externes[index_seed, i_extern] = alphas_optimaux

                print(f"  → Alphas optimaux — médiane log10 : {np.median(np.log10(alphas_optimaux)):.2f}")

                # ── ÉVALUATION EXTERNE ───────────────────────────────────────────────
                test_masque = groupes == session_test

                X_scaled_train_final, X_scaled_test, Y_scaled_train_final, Y_scaled_test = self._scaler_X_Y(
                    X, Y, train_val_masque, test_masque
                )

                modele_final = RidgeCV(
                    alphas=alphas_optimaux,
                    alpha_per_target=True,
                    cv=None,
                )

                modele_final.fit(X_scaled_train_final, Y_scaled_train_final)

                Y_pred_test = modele_final.predict(X_scaled_test)
                r2_test = r2_score(Y_scaled_test, Y_pred_test, multioutput="raw_values")

                r2_tous_les_tests[index_seed, i_extern] = r2_test
                print(f"  → R² max : {np.max(r2_test):.5f}")

                del modele_final, Y_pred_test
                gc.collect()

        # AGRÉGATION
        r2_moyen = np.mean(r2_tous_les_tests, axis=(0, 1))
        r2_variance_inter_folds = np.mean(np.std(r2_tous_les_tests, axis=1))
        r2_variance_inter_tests = np.mean(np.std(r2_tous_les_tests, axis=0))
        alphas_tous_externes_moyen = np.mean(alphas_tous_externes, axis=0)

        tsnr = Y.mean(axis=0) / (Y.std(axis=0) + 1e-8)

        print("Y shape :", Y.shape)
        print("Y min :", np.min(Y))
        print("Y max :", np.max(Y))
        print("Y mean :", np.mean(Y))
        print("Y:", Y)

        return r2_moyen, r2_variance_inter_folds, r2_variance_inter_tests, r2_tous_les_tests, alphas_tous_externes_moyen, tsnr


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

    def plot_r2_distribution(self, r2_moyen, suffix=""):
        fig, ax = plt.subplots(figsize=(10, 5))

        sns.kdeplot(r2_moyen, ax=ax, fill=True, alpha=0.5, linewidth=2, color="#2166ac")

        ax.axvline(0, color="black", linestyle="--", linewidth=1)
        ax.axvline(np.mean(r2_moyen), color="red", linestyle="--", linewidth=1,
                   label=f"Moyenne : {np.mean(r2_moyen):.4f}")
        ax.axvline(np.median(r2_moyen), color="orange", linestyle="--", linewidth=1,
                   label=f"Médiane : {np.median(r2_moyen):.4f}")

        ax.set_xlabel("R²")
        ax.set_ylabel("Densité")
        unite = "voxels" if self.flag_precision_voxel else "parcelles"
        ax.set_title(f"Distribution des R² — {self.subject} / {self.layer} ({unite})")
        ax.legend()
        plt.tight_layout()

        chemins = self.get_path_file_by_plateform(self.plateforme)
        nom_fichier = f"r2_distribution_{self.subject}_{self.layer}{suffix}.png"
        chemin_sortie = chemins.root_encoding / "output" / nom_fichier
        plt.savefig(chemin_sortie, dpi=300)
        plt.close()
        print(f"Distribution R² sauvegardée : {chemin_sortie}")


    def plot_ROImask_histogram(self, scores_finaux, liste_ROI):
        """Trace un boxplot des R² par ROI (voxelwise uniquement) et l'enregistre en HTML."""
        chemins = self.get_path_file_by_plateform(self.plateforme)
        fichier_ROImask = chemins.chemin_ROImask

        ROIs_noms = []
        ROIs_vecteur = []

        if self.flag_precision_voxel:
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
                fig.update_traces(
                    marker=dict(opacity=0.3, size=3),
                )
                fig.write_html(str(chemins.root_encoding / "output" / f"ROImask_{self.subject}.html"))
            print("Histogramme ROI sauvegardé :", str(chemins.root_encoding / "output" / f"ROImask_{self.subject}.html"))
            return
        else:
            return "Pas en voxel"

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
            hue_params = {"hue": "fold", "multiple": "stack", "palette": "tab20"}
            titre = "Distribution des alphas par fold"

        # Limites et ticks communs
        xlim_min = log10_valeurs.min() - step / 2
        xlim_max = log10_valeurs.max() + step / 2
        ticks_visibles = log10_grille[(log10_grille >= xlim_min) & (log10_grille <= xlim_max)]

        # Figure
        unite = "voxels" if self.flag_precision_voxel else "parcelles"
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.lineplot(data=df, x="log10_alpha", bins=bins, shrink=0.8, ax=ax, **hue_params)
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

    def brain_mapping_tsnr(self, suffix=""):
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
        alphas = np.logspace(-1, 10, 12)
        # ── Alignement normal ────────────────────────────────────────────────
        ridge = RidgeRegression(
            plateforme, SUB, LAYER,
            flag_delai_bold_brute, centrage_donne_temps,
            flag_precision_voxel, ROImask_flag, randomize_flag=False
        )

        print("\n[TEST] nested_cross_validation")
        r2_moyen, r2_variance_inter_folds, r2_variance_inter_tests, r2_tous_les_tests, alphas_tous_externes_moyen, tsnr = ridge.nested_cross_validation(alphas)

        alphas_moyens = 10 ** np.mean(np.log10(alphas_tous_externes_moyen), axis=0)

        ridge.brain_mapping_r2(r2_moyen,    suffix="_nested_moyen")
        print(" Variance inter-tests : ", r2_variance_inter_tests)
        print(" Variance inter-folds : ", r2_variance_inter_folds)
        ridge.plot_alphas_histogram(alphas_fold=alphas_tous_externes_moyen, grille_alphas=alphas, suffix="_nested_folds")
        ridge.plot_alphas_histogram(alphas_fold=None, grille_alphas=alphas, alphas_finaux=alphas_moyens, suffix="_nested_moyen")
        ridge._brain_mapping_generique(tsnr, nom_carte="TSNR", cmap="Blues",
                                treshold=0.0, echelle_log=False,
                                vmin=0, vmax=np.max(tsnr),
                                suffix="_nested")
        ridge.plot_r2_distribution(r2_moyen, suffix="_nested")
        
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
