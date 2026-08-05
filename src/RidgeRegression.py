"""
Régression Ridge pour l'encodage cérébral THINGS memory.
Entraîne une Ridge (grille d'alphas balayée manuellement) par couche et évalue la prédiction.
"""
from dataclasses import dataclass
import gc
from pathlib import Path
import warnings

import h5py
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
from nilearn.plotting import plot_stat_map
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from GroupShuffleSplitSession import GroupShuffleSplitSession
from TribeHDF5Normalization import TribeHDF5Normalization

# Ignore spécifiquement les avertissements de matrices mal conditionnées
warnings.filterwarnings(action='ignore', category=LinAlgWarning)

matplotlib.use('Agg')

DPI_FIGURES = 300
T_TRIBE_S = 0.5
TR_IRMF_S = 1.49
SEUIL_AFFICHAGE_BRAIN_MAP = 0.01
SEUIL_DETECTION_Y_STANDARDISE = 0.05

# Protocole "one cycle" (découpage par blocs de sessions)
NB_SESSIONS_TOTAL = 36
SESSIONS_TEST_ONE_CYCLE = (14, 15, 16)
BUFFER_TEST_ONE_CYCLE = (13, 17)

# Blocs de Validation et leurs buffers, donnés explicitement par le protocole
# (bornes et ordre exacts — pas dérivés d'une règle de tuilage générique).
FOLDS_VALIDATION_ONE_CYCLE = (
    {"validation": (1, 2, 3), "buffer": (4,)},
    {"validation": (5, 6, 7), "buffer": (4, 8)},
    {"validation": (10, 11, 12), "buffer": (9, 13)},
    {"validation": (18, 19, 20), "buffer": (17, 21)},
    {"validation": (22, 23, 24), "buffer": (21, 25)},
    {"validation": (26, 27, 28), "buffer": (25, 29)},
    {"validation": (30, 31, 32), "buffer": (29, 33)},
    {"validation": (34, 35, 36), "buffer": (33,)},
)

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
                    t_Tribe_s=T_TRIBE_S,
                    TR_irmf_s=TR_IRMF_S,
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
            if abs(float(np.mean(Y))) < SEUIL_DETECTION_Y_STANDARDISE and abs(float(np.std(Y)) - 1) < SEUIL_DETECTION_Y_STANDARDISE:
                print(
                    "ATTENTION : Y semble déjà standardisé en amont (moyenne globale "
                    f"={np.mean(Y):.2e}, écart-type={np.std(Y):.3f}). Le TSNR calculé ici "
                    "(mean/std sur un signal déjà z-scoré) est proche de 0 par construction "
                    "et ne reflète PAS la qualité du signal brut. Il faut le calculer sur le "
                    "BOLD non standardisé (avant l'extraction des séries temporelles), pas ici."
                )

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

    def nested_cross_validation_full_manuel(self, grille_alphas, n_folds=5, test_size=0.2, seed=None):
        """Validation croisée imbriquée 100% manuelle, jumeau de `nested_cross_validation_ridgecv_loo`.

        Même split externe par session (GroupShuffleSplitSession) et même principe
        (un alpha optimal par voxel/parcelle), mais la sélection d'alpha en boucle
        interne est faite à la main : `LeaveOneGroupOut` (une session isolée à la
        fois) + refit d'un `Ridge` pour chaque alpha de la grille, sur chaque fold
        interne. L'alpha final par voxel est choisi en moyennant d'abord les
        courbes R²(alpha) sur tous les folds internes, puis en prenant l'argmax
        de cette courbe moyennée (plus stable qu'une moyenne d'argmax bruités,
        cf. commentaire plus bas).

        Attention : coûteux. Pour n_folds folds externes × n sessions internes ×
        len(grille_alphas) alphas, ça fait n_folds × n × len(grille_alphas) refits
        de `Ridge()` (~5000 fits pour une config typique 10×25×20), chacun
        recalculant une décomposition depuis zéro. C'est l'unique raison d'être
        de `nested_cross_validation_ridgecv_loo` : remplacer ce triple `for` par
        `RidgeCV(cv=None)`, qui calcule l'équivalent analytiquement en un seul
        fit — mais uniquement disponible en Leave-One-Out (par timepoint), pas en
        Leave-One-Group-Out (par session), d'où la seule différence algorithmique
        entre les deux jumeaux.
        """
        X, Y, groupes, TSNR = self._selection_X_Y()
        n_features = Y.shape[1]

        # 1. Définition du splitter externe
        outer_cv = GroupShuffleSplitSession(n_splits=n_folds, test_size=test_size, random_state=seed)

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)
        best_alphas_inner_toutes_folds = []

        # 2. BOUCLE EXTERNE : Évaluation de la stabilité du modèle
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y, groupes)):
            print(f"  -> Début du Fold externe {i + 1}/{n_folds}...")

            X_train, Y_train, groupes_train = X[train_idx], Y[train_idx], groupes[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            inner_cv = LeaveOneGroupOut()
            inner_splits = list(inner_cv.split(X_train, Y_train, groups=groupes_train))

            n_inner_folds = len(inner_splits)
            r2_par_alpha_cumul = np.zeros((len(grille_alphas), n_features), dtype=np.float64)

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

                # Diagnostic uniquement : meilleur alpha de CE fold interne, par voxel
                # (bruité par construction, sert à visualiser la variabilité inter-sessions,
                # ne sert plus à calculer l'alpha final)
                best_indices_fold = np.argmax(r2_par_alpha, axis=0)
                best_alphas_inner_toutes_folds.append(grille_alphas[best_indices_fold])

                r2_par_alpha_cumul += r2_par_alpha

            # Sélection finale : on moyenne d'abord les courbes R²(alpha) sur les folds
            # internes, puis on prend l'argmax par voxel sur cette courbe moyennée.
            # (moyenner des argmax bruités - via la moyenne géométrique - amplifie les
            # cas où un seul fold interne bascule au bord de la grille)
            r2_par_alpha_moyen = r2_par_alpha_cumul / n_inner_folds
            best_indices = np.argmax(r2_par_alpha_moyen, axis=0)
            alpha_optimal = grille_alphas[best_indices]
            alphas_tous_externes[i, :] = alpha_optimal

            # Standardisation du set externe
            scaler_X = StandardScaler()
            X_train_scaled = scaler_X.fit_transform(X_train)
            X_test_scaled = scaler_X.transform(X_test)

            scaler_Y = StandardScaler()
            Y_train_scaled = scaler_Y.fit_transform(Y_train)
            Y_test_scaled = scaler_Y.transform(Y_test)

            ridge_final = Ridge(alpha=alpha_optimal)
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
        best_alphas_inner = np.array(best_alphas_inner_toutes_folds)

        return r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, best_alphas_inner, TSNR

    def nested_cross_validation_ridgecv_loo(self, grille_alphas, n_folds=5, test_size=0.2, seed=None):
        """Jumeau sklearn-natif de `nested_cross_validation_full_manuel`.

        Même split externe par session (GroupShuffleSplitSession) et même principe
        (un alpha optimal par voxel/parcelle), mais la boucle interne manuelle
        (LeaveOneGroupOut + refit d'un Ridge pour chaque alpha de la grille) est
        remplacée par RidgeCV(cv=None), qui sélectionne l'alpha via un Leave-One-Out
        efficace calculé analytiquement (pas de refit par échantillon ni par alpha).
        C'est l'unique différence algorithmique avec la version manuelle : LOO
        (par timepoint) au lieu de LOGO (par session).

        Attention (voir évaluation) : le LOO ignore la structure de session/
        autocorrélation temporelle que LOGO respectait explicitement pour la
        sélection d'alpha — les alphas et R² obtenus ne sont donc pas strictement
        comparables scientifiquement à ceux de `nested_cross_validation_full_manuel`, seulement
        comparables en termes de mécanique/temps de calcul.
        """
        X, Y, groupes, TSNR = self._selection_X_Y()
        n_features = Y.shape[1]

        outer_cv = GroupShuffleSplitSession(n_splits=n_folds, test_size=test_size, random_state=seed)

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)

        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y, groupes)):
            print(f"  -> Début du Fold externe {i + 1}/{n_folds}...")

            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            # StandardScaler() sur X : fit sur train, transform sur test (Pipeline = 0 fuite).
            # RidgeCV(cv=None) : Leave-One-Out natif et efficace (raccourci algébrique,
            # cf. sklearn.linear_model.RidgeCV). alpha_per_target=True : un alpha par
            # voxel, uniquement compatible avec cv=None (LOO).
            # TransformedTargetRegressor : standardise Y_train, entraîne dessus, et
            # dé-standardise automatiquement les prédictions à l'inverse_transform.
            modele = TransformedTargetRegressor(
                regressor=make_pipeline(
                    StandardScaler(),
                    RidgeCV(alphas=grille_alphas, alpha_per_target=True, cv=None, scoring="r2"),
                ),
                transformer=StandardScaler(),
            )
            modele.fit(X_train, Y_train)

            ridgecv_ajuste = modele.regressor_.named_steps["ridgecv"]
            alphas_tous_externes[i, :] = ridgecv_ajuste.alpha_

            # Prédictions déjà ramenées à l'échelle d'origine par TransformedTargetRegressor
            Y_pred = modele.predict(X_test)
            r2_score_fold = r2_score(Y_test, Y_pred, multioutput='raw_values')
            r2_tous_les_tests[i, :] = r2_score_fold
            print(f"-> R2 mean : {np.mean(r2_score_fold)}")
            print(f"-> R2 max : {np.max(r2_score_fold)}")

            # Nettoyage mémoire
            del modele, Y_pred
            gc.collect()

        r2_moyen = np.mean(r2_tous_les_tests, axis=0)
        r2_variance_inter_folds = np.var(r2_tous_les_tests, axis=0)
        alphas_tous_externes_moyen = np.mean(alphas_tous_externes, axis=0)

        return r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, TSNR

    def _generer_folds_one_cycle(self):
        """Construit les folds Train/Validation/Test/Buffer du protocole 'one cycle'.

        Règles (sessions numérotées 1..36) :
        - Test FIXE, identique pour tous les folds : {14, 15, 16}.
        - Buffer autour du Test, toujours exclu (jamais Train ni Validation) : {13, 17}.
        - Validation et buffer associé : donnés explicitement par `FOLDS_VALIDATION_ONE_CYCLE`
          (8 blocs, bornes et buffers exacts du protocole — pas dérivés d'une règle
          générique de tuilage).
        - Train : tout le reste (ni Test, ni buffer Test, ni Validation, ni buffer
          Validation, pour ce fold).
        """
        sessions_test = set(SESSIONS_TEST_ONE_CYCLE)
        buffer_test = set(BUFFER_TEST_ONE_CYCLE)
        toutes_sessions = set(range(1, NB_SESSIONS_TOTAL + 1))

        folds = []
        for spec in FOLDS_VALIDATION_ONE_CYCLE:
            validation = list(spec["validation"])
            buffer_validation = set(spec["buffer"])
            exclues = sessions_test | buffer_test | set(validation) | buffer_validation
            train = sorted(toutes_sessions - exclues)
            folds.append({
                "train": train,
                "validation": validation,
                "test": sorted(sessions_test),
                "buffer": sorted(buffer_test | buffer_validation),
            })
        return folds

    def nested_cross_validation_one_cycle(self, grille_alphas):
        """Validation croisée par blocs de sessions ('one cycle'), protocole type
        CNeuroMod-THINGS.

        Contrairement aux deux autres jumeaux (`nested_cross_validation_full_manuel`,
        `nested_cross_validation_ridgecv_loo`), le split n'est pas aléatoire : le
        Test est FIXE (sessions 14-16) et identique pour tous les folds, jamais
        utilisé pour choisir l'alpha. Chaque fold ne comporte qu'UN SEUL split
        Train → Validation (pas de sous-CV interne) pour sélectionner l'alpha optimal
        par voxel ; le modèle final est ensuite réentraîné sur Train+Validation et
        évalué une fois sur le Test fixe. Des sessions tampons ("buffer") isolent le
        Test et chaque bloc de Validation de leurs voisines immédiates pour limiter
        la fuite due à l'autocorrélation temporelle inter-sessions.

        Voir `_generer_folds_one_cycle` pour le détail du découpage.
        """
        X, Y, groupes, TSNR = self._selection_X_Y()
        n_features = Y.shape[1]

        folds = self._generer_folds_one_cycle()
        n_folds = len(folds)

        print(f"  -> {n_folds} folds 'one cycle' générés :")
        for i, fold in enumerate(folds):
            print(f"     Fold {i + 1} | Validation={fold['validation']} | Buffer={fold['buffer']} | Train={len(fold['train'])} sessions")

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)

        for i, fold in enumerate(folds):
            print(f"  -> Début du Fold {i + 1}/{n_folds} (Validation={fold['validation']})...")

            masque_train = np.isin(groupes, fold["train"])
            masque_val = np.isin(groupes, fold["validation"])
            masque_test = np.isin(groupes, fold["test"])

            if not masque_val.any() or not masque_test.any():
                print(f"     Fold {i + 1} ignoré : Validation ou Test vide pour {self.subject} (sessions manquantes dans les données).")
                continue

            X_train, Y_train = X[masque_train], Y[masque_train]
            X_val, Y_val = X[masque_val], Y[masque_val]
            X_test, Y_test = X[masque_test], Y[masque_test]

            # Sélection de l'alpha optimal par voxel sur l'unique split Train -> Validation
            # (0 fuite : standardisation fit sur Train seul)
            scaler_X_selection = StandardScaler()
            X_train_scaled_selection = scaler_X_selection.fit_transform(X_train)
            X_val_scaled = scaler_X_selection.transform(X_val)

            scaler_Y_selection = StandardScaler()
            Y_train_scaled_selection = scaler_Y_selection.fit_transform(Y_train)
            Y_val_scaled = scaler_Y_selection.transform(Y_val)

            r2_par_alpha = np.zeros((len(grille_alphas), n_features))
            for a_idx, alpha in enumerate(grille_alphas):
                ridge_selection = Ridge(alpha=alpha)
                ridge_selection.fit(X_train_scaled_selection, Y_train_scaled_selection)
                Y_val_pred = ridge_selection.predict(X_val_scaled)
                r2_par_alpha[a_idx, :] = r2_score(Y_val_scaled, Y_val_pred, multioutput='raw_values')

            alpha_optimal = grille_alphas[np.argmax(r2_par_alpha, axis=0)]
            alphas_tous_externes[i, :] = alpha_optimal

            # Réentraînement final sur Train + Validation, évaluation sur le Test fixe
            X_train_val = np.concatenate([X_train, X_val], axis=0)
            Y_train_val = np.concatenate([Y_train, Y_val], axis=0)

            scaler_X = StandardScaler()
            X_train_val_scaled = scaler_X.fit_transform(X_train_val)
            X_test_scaled = scaler_X.transform(X_test)

            scaler_Y = StandardScaler()
            Y_train_val_scaled = scaler_Y.fit_transform(Y_train_val)
            Y_test_scaled = scaler_Y.transform(Y_test)

            ridge_final = Ridge(alpha=alpha_optimal)
            ridge_final.fit(X_train_val_scaled, Y_train_val_scaled)
            Y_pred_scaled = ridge_final.predict(X_test_scaled)

            r2_score_fold = r2_score(Y_test_scaled, Y_pred_scaled, multioutput='raw_values')
            r2_tous_les_tests[i, :] = r2_score_fold
            print(f"-> R2 mean : {np.mean(r2_score_fold)}")
            print(f"-> R2 max : {np.max(r2_score_fold)}")

            del ridge_final, Y_pred_scaled
            gc.collect()

        r2_moyen = np.mean(r2_tous_les_tests, axis=0)
        r2_variance_inter_folds = np.var(r2_tous_les_tests, axis=0)
        alphas_tous_externes_moyen = np.mean(alphas_tous_externes, axis=0)

        return r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, TSNR

    def _sauvegarder_figure(self, figure_sauvegardable, nom_fichier, message, **kwargs_savefig):
        """Sauvegarde une figure (matplotlib Figure ou display Nilearn) dans output/ et affiche un message."""
        chemins = self.get_path_file_by_plateform(self.plateforme)
        chemin_sortie = chemins.root_encoding / "output" / nom_fichier
        figure_sauvegardable.savefig(chemin_sortie, dpi=DPI_FIGURES, **kwargs_savefig)
        print(f"{message} : {chemin_sortie}")
        return chemin_sortie

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

        nom_fichier = f"r2_distribution_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Distribution R² sauvegardée")
        plt.close(fig)
        return chemin_sauvegarde

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
                chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "ROImask sauvegardé", bbox_inches="tight")
                plt.close(fig)
                return chemin_sauvegarde
        else:
            print("ROImask ignoré : analyse disponible uniquement en précision voxel (flag_precision_voxel=True).")
            return None

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

        nom_fichier = f"r2_threshold_{self.subject}_{self.layer}_{unite}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Threshold R² sauvegardé")
        plt.close(fig)
        return chemin_sauvegarde

    def plot_accuracy(self, r2_tous_les_tests, suffix=""):
        """Barres mean/median/top-10% pour UN sujet (pas d'agrégation multi-sujets :
        un seul appel = un seul sujet, une seule figure)."""
        n_features = r2_tous_les_tests.shape[1]
        n_folds = r2_tous_les_tests.shape[0]

        # Une valeur par fold externe pour que seaborn puisse tracer une barre d'erreur (écart-type inter-folds) :
        # la variabilité entre folds externes est précisément ce que la nested CV permet
        # d'estimer.
        means_par_fold = np.mean(r2_tous_les_tests, axis=1)
        medians_par_fold = np.median(r2_tous_les_tests, axis=1)
        seuils_top10_par_fold = np.percentile(r2_tous_les_tests, 90, axis=1)
        top10_par_fold = np.array([
            np.mean(fold[fold >= seuil])
            for fold, seuil in zip(r2_tous_les_tests, seuils_top10_par_fold)
        ])

        # Format long (une ligne par fold externe × métrique) : errorbar="sd" dans
        # sns.barplot calcule alors automatiquement moyenne + écart-type inter-folds.
        df = pd.DataFrame({
            "Métrique": ["mean"] * n_folds + ["median"] * n_folds + ["top-10% mean"] * n_folds,
            "R2": np.concatenate([means_par_fold, medians_par_fold, top10_par_fold]),
        })

        fig, ax = plt.subplots(figsize=(6, 6))

        # Ajout de la grille légère en arrière-plan (comme sur l'image)
        ax.grid(True, axis='both', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)  # S'assure que la grille reste derrière les barres

        # Couleurs fidèles à votre image de référence
        palette = {"mean": "#0072B2", "median": "#56B4E9", "top-10% mean": "#E69F00"}

        sns.barplot(
            data=df, x="Métrique", y="R2", hue="Métrique",
            palette=palette, ax=ax, errorbar="sd", capsize=0.1, legend=False,
        )

        # Annotation automatique et propre des valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

        # Nettoyage de l'esthétique (retrait des bordures haut et droite)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_xlabel("")
        ax.set_ylabel("R² (raw)", fontsize=12)

        # Titre centré et en gras
        ax.set_title(f"Accuracy — {self.subject} / {self.layer} (n={n_features:,}, {n_folds} folds)", fontsize=14, fontweight='bold')

        plt.tight_layout()

        nom_fichier = f"accuracy_{self.subject}_{self.layer}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Accuracy sauvegardée")
        plt.close(fig)
        return chemin_sauvegarde


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
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Histogramme alphas sauvegardé")
        plt.close(fig)
        return chemin_sauvegarde

    def _brain_mapping_generique(self, donnees, nom_carte, cmap, treshold=SEUIL_AFFICHAGE_BRAIN_MAP, echelle_log=False, vmin = None, vmax = None, suffix=""):
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

        nom_fichier = f"brain_map_{self.subject}_{self.layer}_{nom_carte}_{unite}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(display, nom_fichier, "Carte cérébrale sauvegardée")
        display.close()
        plt.close(fig)
        return chemin_sauvegarde

    def brain_mapping_r2(self, scores_r2, noms_parcelles=None, suffix=""):
        """Affiche le résumé des R² et enregistre la carte cérébrale correspondante."""
        self.print_scores(scores_r2, noms_parcelles)
        chemin_sauvegarde = self._brain_mapping_generique(scores_r2, nom_carte="R2", cmap="YlOrRd", treshold=SEUIL_AFFICHAGE_BRAIN_MAP, echelle_log=False, vmin=0, vmax=np.max(scores_r2), suffix=suffix)
        return chemin_sauvegarde

    def brain_mapping_alphas(self, alphas_tous_les_lots, suffix=""):
        """Enregistre la carte cérébrale des alphas optimaux (échelle log10)."""
        # treshold=0 : la donnée affichée est log10(alpha),
        chemin_sauvegarde = self._brain_mapping_generique(alphas_tous_les_lots, nom_carte="Alphas", cmap="YlOrRd", treshold=0, echelle_log=True, suffix=suffix)
        return chemin_sauvegarde

    def brain_mapping_tsnr(self, tsnr, suffix=""):
        """Enregistre la carte cérébrale correspondante."""
        # évite que les valeurs extrêmes écrasent la colorbar
        chemin_sauvegarde = self._brain_mapping_generique(tsnr, nom_carte="TSNR", cmap="Blues", treshold=0.0, echelle_log=False,vmin=0,vmax=np.percentile(tsnr, 95),suffix=suffix,)
        return chemin_sauvegarde

    def regrouper_figures_dans_une_planche(self, nom_methode, liste_chemins_figures, nombre_de_colonnes=3):
        """Assemble toutes les figures PNG déjà sauvegardées pour UNE méthode de
        validation croisée dans une seule image PNG (une "planche"), au lieu d'avoir
        un fichier séparé par figure.

        `liste_chemins_figures` est la liste des chemins renvoyés par les appels à
        brain_mapping_r2, brain_mapping_alphas, plot_accuracy, etc. Certains appels
        peuvent renvoyer None (par exemple ROImask quand flag_precision_voxel est
        False) : ces entrées sont ignorées ici.
        """
        # Étape 1 : on ne garde que les chemins réellement produits (pas les None).
        chemins_valides = []
        for chemin_figure in liste_chemins_figures:
            if chemin_figure is not None:
                chemins_valides.append(chemin_figure)

        nombre_de_figures = len(chemins_valides)
        if nombre_de_figures == 0:
            print(f"Aucune figure à regrouper pour {nom_methode}.")
            return None

        # Étape 2 : on calcule le nombre de lignes nécessaires pour ranger toutes
        # les figures dans une grille de `nombre_de_colonnes` colonnes.
        nombre_de_lignes = nombre_de_figures // nombre_de_colonnes
        reste_figures = nombre_de_figures % nombre_de_colonnes
        if reste_figures != 0:
            nombre_de_lignes = nombre_de_lignes + 1

        # Étape 3 : on crée la grande figure qui contiendra une grille de sous-figures.
        largeur_par_case = 6
        hauteur_par_case = 6
        largeur_totale = largeur_par_case * nombre_de_colonnes
        hauteur_totale = hauteur_par_case * nombre_de_lignes
        figure_planche, grille_axes = plt.subplots(
            nombre_de_lignes, nombre_de_colonnes,
            figsize=(largeur_totale, hauteur_totale),
        )

        # Étape 4 : on parcourt chaque case de la grille, ligne par ligne puis
        # colonne par colonne, et on y affiche l'image correspondante si elle existe.
        index_figure_courante = 0
        for numero_ligne in range(nombre_de_lignes):
            for numero_colonne in range(nombre_de_colonnes):

                # Récupération de l'axe (la "case") correspondant à cette position,
                # en tenant compte du fait que matplotlib simplifie la forme du
                # tableau d'axes quand il n'y a qu'une seule ligne ou une seule colonne.
                if nombre_de_lignes == 1 and nombre_de_colonnes == 1:
                    axe_courant = grille_axes
                elif nombre_de_lignes == 1:
                    axe_courant = grille_axes[numero_colonne]
                elif nombre_de_colonnes == 1:
                    axe_courant = grille_axes[numero_ligne]
                else:
                    axe_courant = grille_axes[numero_ligne, numero_colonne]

                if index_figure_courante < nombre_de_figures:
                    chemin_image = chemins_valides[index_figure_courante]
                    image_chargee = plt.imread(chemin_image)
                    axe_courant.imshow(image_chargee)
                    nom_court = Path(chemin_image).stem
                    axe_courant.set_title(nom_court, fontsize=8)

                # On masque toujours les axes (case vide ou pas) pour un rendu propre.
                axe_courant.axis("off")

                index_figure_courante = index_figure_courante + 1

        titre_planche = f"{nom_methode} — {self.subject} / {self.layer}"
        figure_planche.suptitle(titre_planche, fontsize=16, fontweight="bold")
        plt.tight_layout()

        nom_fichier_planche = f"planche_{nom_methode}_{self.subject}_{self.layer}.png"
        chemin_sauvegarde = self._sauvegarder_figure(figure_planche, nom_fichier_planche, "Planche de figures sauvegardée")
        plt.close(figure_planche)

        return chemin_sauvegarde

    def generer_toutes_les_figures(self, nom_methode, resultats, grille_alphas, noms_parcelles=None):
        """Génère et sauvegarde l'ensemble des figures standard pour UNE méthode de
        validation croisée (appelée une fois par méthode : full_manuel, ridgecv_loo,
        one_cycle...). `resultats` est le tuple renvoyé par la méthode
        `nested_cross_validation_*` correspondante :
        - 7 éléments pour `nested_cross_validation_full_manuel` (avec `best_alphas_inner`,
          le diagnostic de sous-CV interne — LOGO — que les deux autres n'ont pas) ;
        - 6 éléments pour `nested_cross_validation_ridgecv_loo` et
          `nested_cross_validation_one_cycle` (pas de sous-CV interne).

        Retourne un petit résumé (dict) réutilisable pour comparer les méthodes entre
        elles dans le `__main__`.
        """
        if len(resultats) == 7:
            r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, _, best_alphas_inner, tsnr = resultats
        else:
            r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, _, tsnr = resultats
            best_alphas_inner = None

        suffix = f"_{nom_methode}"
        # Moyenne géométrique sur les folds (les alphas s'étalent sur plusieurs décades)
        alphas_moyens = 10 ** np.mean(np.log10(alphas_tous_externes), axis=0)

        print(f"\n[FIGURES] {nom_methode} — Variance inter-folds moyenne : {np.mean(r2_variance_inter_folds):.6f}")

        # Liste explicite des chemins de chaque figure produite pour cette méthode :
        # elle sert ensuite à assembler toutes ces figures dans une seule planche.
        liste_chemins_figures = []

        # 1. brain_r2_map + 2. alpha_map (+ TSNR, cohérent avec les deux autres cartes)
        print(" -> Cartes cérébrales (R², Alphas, TSNR)...")
        chemin_figure = self.brain_mapping_r2(r2_moyen, noms_parcelles, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.brain_mapping_alphas(alphas_moyens, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        #chemin_figure = self.brain_mapping_tsnr(tsnr, suffix=suffix)
        #liste_chemins_figures.append(chemin_figure)

        # 3. histogrammes des alphas par fold + 4. moyenne des alphas
        print(" -> Histogrammes des alphas...")
        chemin_figure = self.plot_alphas_histogram(alphas_fold=alphas_tous_externes, grille_alphas=grille_alphas, suffix=f"{suffix}_folds")
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.plot_alphas_histogram(alphas_fold=None, grille_alphas=grille_alphas, alphas_finaux=alphas_moyens, suffix=f"{suffix}_moyen")
        liste_chemins_figures.append(chemin_figure)

        # 5. visualisation des alphas internes (uniquement si la méthode en produit)
        if best_alphas_inner is not None:
            chemin_figure = self.plot_alphas_histogram(alphas_fold=best_alphas_inner, grille_alphas=grille_alphas, suffix=f"{suffix}_inner")
            liste_chemins_figures.append(chemin_figure)
        else:
            print(f"  -> Pas d'alphas internes pour {nom_methode} (pas de sous-CV interne).")

        # 6. courbe d'accuracy (single-subject), distributionr2 et r2 treshold
        print(" -> Accuracy et distribution R²...")
        chemin_figure = self.plot_accuracy(r2_tous_les_tests, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.plot_r2_distribution(r2_tous_les_tests, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        chemin_figure = self.plot_r2_threshold(r2_tous_les_tests, suffix=suffix)
        liste_chemins_figures.append(chemin_figure)

        # ROIMask (uniquement en précision voxel)
        if self.flag_precision_voxel:
            print(" -> ROIMask...")
            chemin_figure = self.plot_ROImask_histogram(r2_moyen)
            liste_chemins_figures.append(chemin_figure)
        else:
            print(f"  -> ROIMask ignoré pour {nom_methode} (nécessite flag_precision_voxel=True).")

        # Regroupement de toutes les figures ci-dessus dans un seul fichier PNG.
        print(" -> Assemblage de la planche de figures...")
        self.regrouper_figures_dans_une_planche(nom_methode, liste_chemins_figures)

        return {
            "r2_moyen": r2_moyen,
            "r2_tous_les_tests": r2_tous_les_tests,
            "alphas_moyens": alphas_moyens,
        }

if __name__ == "__main__":
    # Point d'entrée : lance les 3 méthodes de validation croisée pour chaque sujet,
    # génère toutes les figures associées, compare les méthodes entre elles, et lance
    # la baseline randomisée avec la méthode manuelle.

    # --- PARAMÈTRES ---
    plateforme = ["Rorqual", "Mac"]
    plateforme = plateforme[0]

    liste_sujets = ["sub-01", "sub-02", "sub-03", "sub-06"]
    liste_sujets = liste_sujets[2:3]
    LAYER = "encoder_layer7_ffn"

    flag_delai_bold_brute = True
    centrage_donne_temps  = False
    flag_precision_voxel  = False
    ROImask_flag          = False

    for SUB in liste_sujets:
        print(f"\n{'='*60}\n  Sujet : {SUB}\n{'='*60}")

        alphas = np.logspace(-1, 10, 20)

        ridge = RidgeRegression(
            plateforme, SUB, LAYER,
            flag_delai_bold_brute, centrage_donne_temps,
            flag_precision_voxel, ROImask_flag, randomize_flag=False
        )

        # ── Les 3 méthodes de validation croisée ───────────────────────────
        methodes = {
            "full_manuel": lambda: ridge.nested_cross_validation_full_manuel(alphas, n_folds=10, test_size=0.1),
            "ridgecv_loo": lambda: ridge.nested_cross_validation_ridgecv_loo(alphas, n_folds=10, test_size=0.1),
            "one_cycle": lambda: ridge.nested_cross_validation_one_cycle(alphas),
        }

        resume_par_methode = {}
        for nom_methode, executer in methodes.items():
            print(f"\n{'-'*60}\n[MÉTHODE] {nom_methode}\n{'-'*60}")
            resultats = executer()
            resume_par_methode[nom_methode] = ridge.generer_toutes_les_figures(nom_methode, resultats, alphas)

        # ── Comparaison entre méthodes ──────────────────────────────────────
        print(f"\n{'='*60}\n  Comparaison des méthodes — {SUB}\n{'='*60}")
        print(f"{'Méthode':15s} | {'R² moyen':>10s} | {'R² médian':>10s} | {'R² max':>10s}")
        for nom_methode, resume in resume_par_methode.items():
            r2 = resume["r2_moyen"]
            print(f"{nom_methode:15s} | {np.mean(r2):10.4f} | {np.median(r2):10.4f} | {np.max(r2):10.4f}")

        # ── Baseline randomisée (randomization test), méthode manuelle uniquement ──
        print(f"\n{'='*60}\n  Randomization test — {SUB} (nested_cross_validation_full_manuel)\n{'='*60}")
        ridge_random = RidgeRegression(
            plateforme, SUB, LAYER,
            flag_delai_bold_brute, centrage_donne_temps,
            flag_precision_voxel, ROImask_flag, randomize_flag=True
        )
        resultats_random = ridge_random.nested_cross_validation_full_manuel(alphas, n_folds=10, test_size=0.1)
        ridge_random.generer_toutes_les_figures("full_manuel_randomise", resultats_random, alphas)

        print(f"\nTerminé pour le sujet {SUB}. Toutes les figures ont été sauvegardées.")
