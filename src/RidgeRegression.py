"""
Régression Ridge pour l'encodage cérébral THINGS memory.

Entraîne une Ridge (grille d'alphas balayée manuellement) par couche et
évalue la prédiction via trois variantes de validation croisée imbriquée.
"""
from dataclasses import dataclass
import gc
from pathlib import Path
import warnings

import h5py
import numpy as np
import pandas as pd
from scipy.linalg import LinAlgWarning
from scipy.stats import ConstantInputWarning, pearsonr
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from GroupShuffleSplitRun import GroupShuffleSplitRun
from GroupShuffleSplitSession import GroupShuffleSplitSession
from litcoder_folding import create_chunked_folds_trimmed
from TribeHDF5Normalization import TribeHDF5Normalization
from VisualisationResultats import VisualisationResultats

# Ignore spécifiquement les avertissements de matrices mal conditionnées
warnings.filterwarnings(action='ignore', category=LinAlgWarning)

# Idem pour les colonnes constantes rencontrées par `pearsonr` : un voxel plat, ou
# une prédiction plate quand l'alpha retenu est proche de 1e10, donne un écart-type
# nul donc une corrélation indéfinie. scipy renvoie alors NaN (ramené à 0 juste après
# le calcul), ce qui est le comportement voulu — l'avertissement n'apporte rien.
warnings.filterwarnings(action='ignore', category=ConstantInputWarning)

# Idem pour les débordements numpy (divide/overflow/invalid) rencontrés lors des
# fits Ridge avec les plus grands alphas de la grille (jusqu'à 1e10) : ils produisent
# potentiellement des coefficients NaN/Inf pour ces alphas précis, mais n'empêchent
# pas la sélection de l'alpha optimal (leur R² correspondant devient alors très
# négatif ou NaN, donc jamais retenu par argmax).
np.seterr(divide='ignore', over='ignore', invalid='ignore')

T_TRIBE_S = 0.5
TR_IRMF_S = 1.49
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

# Groupes de ROIs voxelwise (noms exacts des datasets dans le fichier
# "<subject>_space-T1w_desc-ROImasks_voxelAnnotations.h5", groupes retinotopy_ROIs / fLoc_ROIs)
ROIS_RETINOTOPIQUES = ("V1", "V2", "V3", "V3a", "V3b", "hV4", "VO1", "VO2", "LO1", "LO2", "TO1", "TO2")
ROIS_CATEGORIELLES = ("faceFFA", "faceOFA", "facepSTS", "bodyEBA", "scenePPA", "sceneOPA", "sceneMPA")

# Réseaux Yeo-7 (noms exacts tels qu'ils apparaissent dans la colonne "name" du
# fichier d'annotations de l'atlas cneuromod26, ex. "7Networks_LH_Vis_1") utilisés
# pour restreindre l'analyse en précision parcelle.
RESEAUX_PARCELLES_VISUEL = ("Vis",)
RESEAUX_PARCELLES_VISUEL_DORSATTN = ("Vis", "DorsAttn")

NOM_FICHIER_ANNOTATIONS_PARCELLES = "tpl-MNI152NLin2009cAsym_atlas-Schaefer2018TianS3NettekovenAsym_desc-1000Parcels7Networks50Subcort128Cereb_parcelAnnotations.tsv"

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
    chemin_annotations_parcelles: Path = None


@dataclass
class ResultatsCV:
    """Sortie commune des quatre méthodes `nested_cross_validation_*`.

    Les champs obligatoires sont produits par toutes les méthodes ; les champs
    optionnels ne le sont que par certaines, et valent None ailleurs. C'est ce qui
    remplace l'ancien dispatch par longueur de tuple côté figures : on teste
    `resultats.best_alphas_inner is not None`, pas `len(resultats) == 7`.

    `VisualisationResultats` lit ces champs par attribut sans importer ce module :
    la dépendance reste à sens unique (RidgeRegression -> VisualisationResultats).

    Attributs :
        r2_moyen : R² moyen par voxel/parcelle, shape (n_features,).
        r2_variance_inter_folds : variance du R² entre folds externes, (n_features,).
        r2_tous_les_tests : R² par fold externe, (n_folds, n_features).
        alphas_tous_externes : alpha retenu par fold externe, (n_folds, n_features).
        alphas_tous_externes_moyen : moyenne arithmétique des alphas, (n_features,).
        TSNR : rapport signal/bruit temporel par voxel/parcelle, (n_features,).
        best_alphas_inner : alphas des folds internes LOGO — `full_manuel` seule.
        pearson_moyen, pearson_variance_inter_folds, pearson_tous_les_tests :
            corrélation de Pearson entre BOLD mesuré et prédit — `one_cycle` seule.
    """

    r2_moyen: np.ndarray
    r2_variance_inter_folds: np.ndarray
    r2_tous_les_tests: np.ndarray
    alphas_tous_externes: np.ndarray
    alphas_tous_externes_moyen: np.ndarray
    TSNR: np.ndarray
    best_alphas_inner: np.ndarray = None
    pearson_moyen: np.ndarray = None
    pearson_variance_inter_folds: np.ndarray = None
    pearson_tous_les_tests: np.ndarray = None

    # Champs indexés par voxel/parcelle, shape (n_features,).
    _CHAMPS_PAR_FEATURE = (
        "r2_moyen", "r2_variance_inter_folds", "alphas_tous_externes_moyen",
        "TSNR", "pearson_moyen", "pearson_variance_inter_folds",
    )
    # Champs indexés (lignes, n_features) : folds externes, ou (folds × sessions)
    # internes pour `best_alphas_inner`. Seule la 2e dimension est restreinte.
    _CHAMPS_PAR_LIGNE = (
        "r2_tous_les_tests", "alphas_tous_externes", "best_alphas_inner",
        "pearson_tous_les_tests",
    )

    def restreindre(self, masque):
        """Renvoie les mêmes résultats limités à un sous-ensemble de voxels/parcelles.

        Chaque voxel est régressé INDÉPENDAMMENT des autres : `StandardScaler`
        normalise colonne par colonne, `alpha_per_target=True` choisit un alpha par
        cible, `r2_score(multioutput='raw_values')` score par colonne, et le découpage
        train/test ne dépend que des sessions/runs, pas du masque. Sélectionner des
        colonnes après coup donne donc exactement ce qu'aurait donné une CV relancée
        sur ce seul sous-ensemble — vérifié à ~1e-16 près sur les deux chemins de
        modèle du projet, cf. `test/test_equivalence_scopes.py`.

        C'est ce qui permet de ne lancer la CV qu'une fois pour tous les scopes,
        au lieu d'une fois par scope (et donc de ne charger les données qu'une fois).

        Args :
            masque : booléen par voxel/parcelle, exprimé dans l'espace de CES
                résultats — pas dans l'espace complet du cerveau si la CV a elle-même
                été restreinte (cf. `masque_relatif`). None = aucune restriction.

        Returns :
            ResultatsCV : nouvelle instance, ou `self` si `masque` est None. Les
                champs optionnels valant None le restent.
        """
        if masque is None:
            return self

        champs = {}
        for nom in self._CHAMPS_PAR_FEATURE:
            valeur = getattr(self, nom)
            champs[nom] = None if valeur is None else valeur[masque]
        for nom in self._CHAMPS_PAR_LIGNE:
            valeur = getattr(self, nom)
            champs[nom] = None if valeur is None else valeur[:, masque]
        return ResultatsCV(**champs)


def scope_disponible(masque_scope, masque_cv):
    """Dit si un scope peut être dérivé des résultats d'une CV, sans la relancer.

    Séparé de `masque_relatif` à dessein : ce dernier renvoie None pour signifier
    « aucune restriction à appliquer », ce qui ne doit surtout pas se confondre avec
    « scope indisponible ». Deux questions distinctes, deux fonctions.

    Args :
        masque_scope : booléen sur l'espace complet, ou None (= tout le cerveau).
        masque_cv : booléen sur l'espace complet ayant servi à la CV, ou None.

    Returns :
        bool : True si le scope est inclus dans ce que la CV a réellement couvert.
            Un scope indisponible n'est pas une erreur, juste une case en moins dans
            la figure de comparaison.
    """
    if masque_cv is None:
        return True                 # la CV couvre tout : n'importe quel scope en dérive
    if masque_scope is None:
        # "Tout le cerveau" ne se dérive pas d'une CV restreinte : l'afficher
        # reviendrait à étiqueter le sous-espace de la CV comme s'il était complet.
        return False
    return bool(np.all(masque_cv[masque_scope]))


def masque_relatif(masque_scope, masque_cv):
    """Exprime un masque de scope dans l'espace des résultats d'une CV restreinte.

    Les masques de scope (`_charger_masque_roi`, `_charger_masque_parcelles`) sont
    définis sur l'espace COMPLET du cerveau, alors que `ResultatsCV.restreindre`
    attend un masque exprimé dans l'espace des résultats. Quand la CV a tourné sur
    tout le cerveau (le cas courant), les deux coïncident ; sinon il faut reprojeter.

    À n'appeler que sur un scope validé par `scope_disponible`.

    Args :
        masque_scope : booléen sur l'espace complet, ou None (= tout le cerveau).
        masque_cv : booléen sur l'espace complet ayant servi à la CV, ou None.

    Returns :
        np.ndarray | None : masque à passer à `ResultatsCV.restreindre`. None
            signifie « aucune restriction », jamais « indisponible ».
    """
    if masque_cv is None:
        return masque_scope
    return masque_scope[masque_cv]


class RidgeRegression:
    """Entraîne et évalue une régression Ridge pour prédire l'activité IRMf
    à partir des activations d'une couche du modèle TRIBE, par sujet."""

    def __init__(self, plateforme, subject, layer, flag_delai_bold_brute, centrage_donne_temps, flag_precision_voxel, ROImask_flag, randomize_flag=False):
        """Initialise la configuration d'un sujet (chemins, options de prétraitement).

        Args :
            plateforme : "Rorqual" (cluster) ou toute autre valeur (poste local/Mac).
            subject : identifiant du sujet, ex. "sub-03".
            layer : nom de la couche TRIBE dont on utilise les activations.
            flag_delai_bold_brute : voir `TribeHDF5Normalization`.
            centrage_donne_temps : voir `TribeHDF5Normalization`.
            flag_precision_voxel : True = timeseries voxelwise, False = parcelles.
            ROImask_flag : Afficher plot ROImask.
            randomize_flag : si True, permute aléatoirement l'ordre des runs de Y
                (baseline de randomisation).
        """
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
        (cluster Rorqual ou poste local) et le niveau de précision (voxel/parcelle).

        Args :
            plateforme : "Rorqual" (cluster) ou toute autre valeur (poste local/Mac).

        Returns :
            CheminsProjet : tous les chemins nécessaires pour ce sujet.
        """
        if plateforme == "Rorqual":
            ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
            ROOT_TIMESERIES = Path("/home/aclaud/links/scratch/things.timeseries")
            # Le dossier des sorties TRIBE ne porte pas le même nom des deux côtés :
            # "hdf5" sur le cluster, "features" en local. Seul écart de chemin entre
            # les deux plateformes, tout le reste se déduit des racines ci-dessus.
            dossier_features = "hdf5"
        else:
            ROOT_ENCODING = Path(__file__).parent.parent
            ROOT_TIMESERIES = ROOT_ENCODING / "data"
            dossier_features = "features"

        chemin_tribe = ROOT_ENCODING / "output" / dossier_features / "things_encoding" / f"{self.subject}.h5"
        chemin_ROImask = ROOT_ENCODING / "data" / "brain_map_subj" / f"{self.subject}_space-T1w_desc-ROImasks_voxelAnnotations.h5"

        chemin_annotations_parcelles = ROOT_ENCODING / "data" / "brain_map_subj" / NOM_FICHIER_ANNOTATIONS_PARCELLES

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

        return CheminsProjet(ROOT_ENCODING, ROOT_TIMESERIES, chemin_tribe, chemin_cneuromod, chemin_atlas, chemin_ROImask, chemin_anatomie, chemin_annotations_parcelles)

    def discover_runs(self, tribe_hdf5=None):
        """Liste les runs disponibles dans le fichier HDF5 TRIBE et fait correspondre
        chacun à sa session/run CNeuroMod et à sa vidéo source.

        Args :
            tribe_hdf5 : fichier HDF5 TRIBE déjà ouvert (évite de le rouvrir si
                l'appelant en a déjà un). Si None, ouvert et fermé ici.

        Returns :
            list[tuple] : un tuple (tribe_ses, tribe_run, chemin_video, cneuromod_ses,
                cneuromod_dataset) par run trouvé.
        """
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
                        chemin_video = chemins.root_encoding / "data" / "things_mp4_vfr" / self.subject / tribe_ses / nom_video

                    runs.append((tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset))
        finally:
            if gere_localement:
                tribe_hdf5.close()

        print(f"{len(runs)} runs trouvés dans {chemins.chemin_tribe.name}")
        return runs

    def create_X_Y_total(self):
        """Construit les matrices X (activations) et Y (signal IRMf) en alignant
        temporellement chaque run, puis les concatène sur l'ensemble des runs.

        Returns :
            tuple : (runs_ok, X, Y, groupes, TSNR)
                - runs_ok (list[str]) : runs traités avec succès ("ses-XXX/run-Y").
                - X, Y (np.ndarray) : activations et signal IRMf concaténés.
                - groupes (np.ndarray) : numéro de session pour chaque échantillon.
                - TSNR (np.ndarray) : rapport signal/bruit temporel par voxel/parcelle.
        """
        chemins = self.get_path_file_by_plateform(self.plateforme)

        X_list, Y_list = [], []
        runs_ok = []
        groupes_list = []
        runs_list = []

        print(f"Traitement des runs pour {self.subject} (TRIBE layer={self.layer})...")
        with h5py.File(chemins.chemin_tribe, "r") as tribe_hdf5, \
                h5py.File(chemins.chemin_cneuromod, "r") as cneuromod_hdf5:

            runs = self.discover_runs(tribe_hdf5=tribe_hdf5)

            for (tribe_ses, tribe_run, chemin_video, cneuromod_ses, cneuromod_dataset) in runs:
                # Vérifier que la vidéo source existe localement
                # Un seul run est en cause, pas la session entière : la règle visait au
                # départ toute la ses-08 de sub-06, avant qu'on identifie que seul son
                # run 6 est mal aligné. Les runs sont nommés "run-1".."run-6" (sans zéro
                # de remplissage), d'où la comparaison à "run-6".
                if self.subject == "sub-06" and cneuromod_ses == "ses-08" and tribe_run == "run-6":
                    print(f"{cneuromod_ses}/{tribe_run} ignoré pour {self.subject} "
                          "(décision manuelle : mauvais alignement spatial après prétraitement).")
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
                runs_list.append(np.full(X_run.shape[0], f"{tribe_ses}/{tribe_run}"))

            print(f"\n{len(runs_ok)} runs traités avec succès")

            if self.randomize_flag:
                rng = np.random.default_rng(42)
                nombre_de_runs = len(Y_list)

                while True:
                    nouvel_ordre = rng.permutation(nombre_de_runs)
                    if not np.any(nouvel_ordre == np.arange(nombre_de_runs)):
                        break

                Y_list = [Y_list[i] for i in nouvel_ordre]
                print(f"Baseline activée : Y_list réordonné aléatoirement ({nombre_de_runs} runs, aucun n'a gardé sa position d'origine)")

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

            # Identifiant de run par échantillon (ex. "ses-014/run-03"), pas exposé
            # dans le tuple retourné (pour ne pas casser les appelants existants) :
            # stocké comme attribut, filtré en parallèle de X/Y/groupes par
            # `_selection_X_Y`, et utilisé pour les diagnostics de composition des
            # folds (cf. `_afficher_composition_runs`).
            self._runs_par_echantillon = np.concatenate(runs_list, axis=0)

            del X_list, Y_list, groupes_list, runs_list

            return runs_ok, X, Y, groupes, TSNR

    def _selection_X_Y(self, sessions_a_exclure=None, masque_roi=None):
        """Construit X, Y, et applique les filtres optionnels (sessions, ROI).

        Args :
            sessions_a_exclure : numéros de session à retirer de X/Y/groupes.
            masque_roi : vecteur booléen par voxel ; si fourni, restreint Y et TSNR
                aux seuls voxels sélectionnés.

        Returns :
            tuple : (X, Y, groupes, TSNR), filtrés selon les arguments ci-dessus.
        """
        runs_ok, X, Y, groupes, TSNR = self.create_X_Y_total()
        if sessions_a_exclure is not None:
            masque = ~np.isin(groupes, sessions_a_exclure)
            X, Y, groupes = X[masque], Y[masque], groupes[masque]
            self._runs_par_echantillon = self._runs_par_echantillon[masque]
        if masque_roi is not None:
            Y = Y[:, masque_roi]
            TSNR = TSNR[masque_roi]
        return X, Y, groupes, TSNR

    def _splitter_externe(self, niveau_split, groupes, n_folds, test_size, seed, n_buffer=1):
        """Choisit le splitter de la boucle EXTERNE et le tableau de groupes associé.

        Args :
            niveau_split : "session" (tirage de sessions entières) ou "run" (LORO
                aléatoire). Aucun des deux ne domine l'autre : la session est plus
                stricte sur les confusions à l'échelle du jour, le run supprime la
                contiguïté temporelle immédiate mais laisse les runs frères de la
                session testée en train (cf. docstring de `GroupShuffleSplitRun`).
            groupes : numéro de session par échantillon (utilisé si "session").
            n_folds, test_size, seed : passés tels quels au splitter.
            n_buffer : nombre de runs écartés de part et d'autre de chaque run de
                test ; ignoré si niveau_split="session". n_buffer=0 conserve la
                taille de train du niveau session, ce qui isole l'effet « runs
                frères » de l'effet « moins de données ».

        Returns :
            tuple : (splitter, groupes_du_split) à donner à `splitter.split(...)`.
        """
        if niveau_split == "run":
            splitter = GroupShuffleSplitRun(n_splits=n_folds, test_size=test_size, random_state=seed, n_buffer=n_buffer)
            return splitter, self._runs_par_echantillon
        if niveau_split == "session":
            splitter = GroupShuffleSplitSession(n_splits=n_folds, test_size=test_size, random_state=seed)
            return splitter, groupes
        raise ValueError(f"niveau_split inconnu : {niveau_split!r} (attendu 'session' ou 'run')")

    def _plages_contigues(self, positions):
        """Condense une liste de positions (int) triables en plages contiguës
        (ex. [0,1,2,3,10,11] -> ["0-3", "10-11"]), pour un affichage résumé au lieu
        de lister chaque TR un par un."""
        positions = np.asarray(sorted(int(p) for p in positions))
        if len(positions) == 0:
            return []
        ruptures = np.where(np.diff(positions) > 1)[0]
        debuts = np.concatenate(([0], ruptures + 1))
        fins = np.concatenate((ruptures, [len(positions) - 1]))
        plages = []
        for d, f in zip(debuts, fins):
            if positions[d] == positions[f]:
                plages.append(f"{positions[d]}")
            else:
                plages.append(f"{positions[d]}-{positions[f]}")
        return plages

    def _afficher_composition_runs(self, groupes, runs, train_idx, test_idx):
        """Affiche, pour chaque session dont des runs se retrouvent à la fois en
        train et en test, le détail des runs de chaque côté (juste les numéros de
        run, la session est déjà donnée par la ligne). Pour un run coupé (présent
        des deux côtés), affiche en plus les plages de TR (position dans le run,
        0 = premier TR acquis) envoyées en train / test / ni l'un ni l'autre
        (TR retirés par `trim_size`), et signale explicitement tout TR qui
        apparaîtrait à la fois en train ET en test (ne devrait structurellement
        jamais arriver : `create_chunked_folds_trimmed` partitionne les chunks
        sans recouvrement — ce contrôle sert à le vérifier plutôt qu'à le supposer).

        Diagnostic pensé pour les splits qui ignorent la structure de session (ex.
        `create_chunked_folds_trimmed`, dont les chunks sont des runs individuels) :
        une session "coupée" signifie que certains de ses runs sont en train et
        d'autres en test, donc une fuite potentielle (même session, même jour,
        même bruit physiologique/drift des deux côtés). Un run "coupé" (TR en
        train et TR en test dans le MÊME run) indique en plus que `chunk_length`
        n'est pas alignée sur les frontières réelles de run (ex. runs de longueur
        variable après normalisation) : le point de coupure tombe alors n'importe
        où au milieu du run, pas à son bord.

        Args :
            groupes : numéro de session par échantillon (retourné par `_selection_X_Y`).
            runs : identifiant de run par échantillon (`self._runs_par_echantillon`,
                aligné sur `groupes` après filtrage par `_selection_X_Y`).
            train_idx, test_idx : indices du fold courant.
        """
        n_samples = len(groupes)
        est_train = np.zeros(n_samples, dtype=bool)
        est_test = np.zeros(n_samples, dtype=bool)
        est_train[train_idx] = True
        est_test[test_idx] = True

        run_numeros = np.array([int(str(r).rsplit("run-", 1)[1]) for r in runs])

        sessions_toutes = sorted(set(int(s) for s in groupes))
        sessions_train = set(int(s) for s in groupes[train_idx])
        sessions_test = set(int(s) for s in groupes[test_idx])
        sessions_melangees = sorted(sessions_train & sessions_test)

        print(f"    Sessions : train {len(sessions_train)}/{len(sessions_toutes)}, "
              f"test {len(sessions_test)}/{len(sessions_toutes)}, "
              f"mélangées train+test (fuite potentielle) {len(sessions_melangees)}/{len(sessions_toutes)}")

        if not sessions_melangees:
            return

        for ses in sessions_melangees:
            masque_ses = groupes == ses
            runs_train = sorted(int(r) for r in set(run_numeros[masque_ses & est_train]))
            runs_test = sorted(int(r) for r in set(run_numeros[masque_ses & est_test]))
            print(f"      ses-{ses:03d} : train={str(runs_train):<18} test={runs_test}")

            for r in sorted(set(runs_train) & set(runs_test)):
                masque_run = masque_ses & (run_numeros == r)
                indices_run = np.where(masque_run)[0]  # ordre d'acquisition (contigu par construction)
                train_run = np.isin(indices_run, train_idx)
                test_run = np.isin(indices_run, test_idx)

                plages_train = self._plages_contigues(np.where(train_run)[0])
                plages_test = self._plages_contigues(np.where(test_run)[0])
                detail = (f"train TR {','.join(plages_train) or '—'} ({int(train_run.sum())})  |  "
                          f"test TR {','.join(plages_test) or '—'} ({int(test_run.sum())})")

                ni_lun_ni_lautre = ~(train_run | test_run)
                if ni_lun_ni_lautre.any():
                    plages_trim = self._plages_contigues(np.where(ni_lun_ni_lautre)[0])
                    detail += f"  |  retiré (trim) TR {','.join(plages_trim)} ({int(ni_lun_ni_lautre.sum())})"

                print(f"        run-{r} ({len(indices_run)} TR) : {detail}")

                chevauchement = train_run & test_run
                if chevauchement.any():
                    plages_dup = self._plages_contigues(np.where(chevauchement)[0])
                    print(f"        !! {int(chevauchement.sum())} TR du run-{r} en train ET en test : {','.join(plages_dup)}")

    def _charger_masque_roi(self, noms_rois):
        """Charge le masque booléen (union) des voxels appartenant à une ou plusieurs
        ROIs, depuis le fichier ROImask du sujet (précision voxel uniquement). La
        recherche parcourt tous les groupes du fichier (retinotopy_ROIs, fLoc_ROIs,
        yeo_ROIs...).

        Args :
            noms_rois : noms de ROIs à combiner (ex. ROIS_RETINOTOPIQUES).

        Returns :
            np.ndarray : masque booléen, une entrée par voxel (True = dans une des
                ROIs demandées).

        Raises :
            ValueError : si une ROI demandée est absente du fichier.
        """
        chemins = self.get_path_file_by_plateform(self.plateforme)

        masque = None
        rois_trouvees = {}
        with h5py.File(chemins.chemin_ROImask, 'r') as fichier:
            for groupe in fichier.keys():
                for nom_roi in fichier[groupe].keys():
                    if nom_roi in noms_rois:
                        vecteur = fichier[groupe][nom_roi][:].astype(bool)
                        masque = vecteur.copy() if masque is None else (masque | vecteur)
                        rois_trouvees[nom_roi] = int(vecteur.sum())

        rois_manquantes = set(noms_rois) - set(rois_trouvees)
        if rois_manquantes:
            raise ValueError(f"ROIs introuvables dans {chemins.chemin_ROImask.name} : {sorted(rois_manquantes)}")

        print(f"Masque ROI ({len(noms_rois)} ROIs) : {int(masque.sum())} voxels au total — détail : {rois_trouvees}")
        return masque

    def _charger_masque_parcelles(self, reseaux):
        """Charge le masque booléen des parcelles appartenant à un ou plusieurs
        réseaux Yeo-7 (précision parcelle uniquement), depuis le fichier
        d'annotations partagé de l'atlas cneuromod26 (`NOM_FICHIER_ANNOTATIONS_PARCELLES`,
        colonne "name", ex. "7Networks_LH_Vis_1" pour le réseau "Vis"). L'ordre des
        lignes du fichier (colonne "index", 1..1134) correspond à l'ordre des
        colonnes des timeseries parcellaires.

        Args :
            reseaux : noms de réseaux à combiner (ex. RESEAUX_PARCELLES_VISUEL_DORSATTN).

        Returns :
            np.ndarray : masque booléen, une entrée par parcelle (True = dans un des
                réseaux demandés).
        """
        chemins = self.get_path_file_by_plateform(self.plateforme)
        annotations = pd.read_csv(chemins.chemin_annotations_parcelles, sep="\t")

        pattern = "|".join(f"_{reseau}_" for reseau in reseaux)
        masque = annotations["name"].str.contains(pattern, regex=True).to_numpy()

        print(f"Masque parcelles ({', '.join(reseaux)}) : {int(masque.sum())} parcelles au total.")
        return masque

    def nested_cross_validation_full_manuel(self, grille_alphas, n_folds=5, test_size=0.2, niveau_split="session", seed=None, masque_roi=None):
        """Validation croisée imbriquée 100% manuelle : sélection d'alpha par voxel/
        parcelle via `LeaveOneGroupOut` (une session isolée à la fois) et refit
        explicite d'un `Ridge` pour chaque alpha de la grille sur chaque fold interne.

        Args :
            grille_alphas : valeurs d'alpha à tester.
            n_folds : nombre de folds externes.
            test_size : part du test externe de chaque fold, exprimée dans l'unité de
                `niveau_split` (proportion si float, nombre de groupes si int).
            niveau_split : "session" ou "run", cf. `_splitter_externe`.
            seed : graine du split externe.
            masque_roi : vecteur booléen par voxel (cf. `_charger_masque_roi`) pour
                restreindre l'analyse à une ROI ; None = cerveau entier.

        Returns :
            ResultatsCV : avec `best_alphas_inner` renseigné (seule méthode à produire
                des alphas de folds internes) et les champs Pearson à None.

        Notes :
            - Jumeau de `nested_cross_validation_ridgecv_loo` : même split externe et
              même principe (un alpha par voxel), mais boucle interne manuelle (LOGO)
              plutôt que LOO analytique
            - La sélection d'alpha utilise un R² poolé (résidus accumulés sur tous les
              folds internes puis un seul calcul de R², jamais une moyenne de R² par
              fold), pour reproduire exactement le mécanisme de `RidgeCV(cv=None)`.
            - `niveau_split` ne change QUE la boucle externe : la boucle interne reste
              un `LeaveOneGroupOut` par session, quelle que soit la valeur choisie.
        """
        X, Y, groupes, TSNR = self._selection_X_Y(masque_roi=masque_roi)
        n_features = Y.shape[1]

        # 1. Définition du splitter externe
        outer_cv, groupes_split = self._splitter_externe(niveau_split, groupes, n_folds, test_size, seed)

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)
        best_alphas_inner_toutes_folds = []

        # 2. BOUCLE EXTERNE : Évaluation de la stabilité du modèle
        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y, groupes_split)):
            print(f"  -> Début du Fold externe {i + 1}/{n_folds}...")

            X_train, Y_train, groupes_train = X[train_idx], Y[train_idx], groupes[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            print(f"    Train : {len(train_idx)} samples, Test : {len(test_idx)} samples")
            self._afficher_composition_runs(groupes, self._runs_par_echantillon, train_idx, test_idx)

            inner_cv = LeaveOneGroupOut()
            inner_splits = list(inner_cv.split(X_train, Y_train, groups=groupes_train))

            # SST du R² poolé : indépendant de l'alpha/fold, calculé une seule fois sur
            # tout Y_train (le LOGO partitionne sans recouvrement).
            Y_train_moyenne = Y_train.mean(axis=0)
            sst_total = np.sum((Y_train - Y_train_moyenne) ** 2, axis=0)

            # Accumulateur du numérateur du R² poolé (SSR), un par alpha.
            ssr_cumul_par_alpha = np.zeros((len(grille_alphas), n_features), dtype=np.float64)

            # On teste chaque fold interne (une session isolée en validation)
            for j, (inner_train_idx, inner_val_idx) in enumerate(inner_splits):

                # Standardisation locale stricte au fold interne (0 fuite)
                scaler_X_inner = StandardScaler()
                X_inner_train_scaled = scaler_X_inner.fit_transform(X_train[inner_train_idx])
                X_inner_val_scaled = scaler_X_inner.transform(X_train[inner_val_idx])

                scaler_Y_inner = StandardScaler()
                Y_inner_train_scaled = scaler_Y_inner.fit_transform(Y_train[inner_train_idx])

                # Unités brutes, requis pour accumuler des résidus cohérents entre folds
                # (chaque fold a son propre scaler_Y_inner).
                Y_inner_val_brut = Y_train[inner_val_idx]

                # R² de CE fold interne uniquement : diagnostic, ne sert pas à choisir l'alpha final.
                r2_par_alpha_ce_fold = np.zeros((len(grille_alphas), n_features))

                for a_idx, alpha in enumerate(grille_alphas):
                    ridge_inner = Ridge(alpha=alpha)
                    ridge_inner.fit(X_inner_train_scaled, Y_inner_train_scaled)
                    Y_inner_pred_scaled = ridge_inner.predict(X_inner_val_scaled)
                    Y_inner_pred_brut = scaler_Y_inner.inverse_transform(Y_inner_pred_scaled)

                    # Numérateur (SSR) du R² poolé, accumulé across folds.
                    ssr_cumul_par_alpha[a_idx, :] += np.sum((Y_inner_val_brut - Y_inner_pred_brut) ** 2, axis=0)
                    r2_par_alpha_ce_fold[a_idx, :] = r2_score(Y_inner_val_brut, Y_inner_pred_brut, multioutput='raw_values')

                best_indices_fold = np.argmax(r2_par_alpha_ce_fold, axis=0)
                best_alphas_inner_toutes_folds.append(grille_alphas[best_indices_fold])

            # R² poolé sur toutes les prédictions internes réunies (plus stable qu'une
            # moyenne de R² calculés séparément par petit fold interne).
            r2_par_alpha_poole = 1 - (ssr_cumul_par_alpha / sst_total)
            best_indices = np.argmax(r2_par_alpha_poole, axis=0)
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
        best_alphas_inner = np.array(best_alphas_inner_toutes_folds)

        return ResultatsCV(
            r2_moyen=r2_moyen,
            r2_variance_inter_folds=r2_variance_inter_folds,
            r2_tous_les_tests=r2_tous_les_tests,
            alphas_tous_externes=alphas_tous_externes,
            alphas_tous_externes_moyen=alphas_tous_externes_moyen,
            TSNR=TSNR,
            best_alphas_inner=best_alphas_inner,
        )

    def nested_cross_validation_ridgecv_loo(self, grille_alphas, n_folds=5, test_size=0.2, niveau_split="session", n_buffer=1, seed=None, masque_roi=None):
        """Jumeau sklearn-natif de `nested_cross_validation_full_manuel` : remplace la
        boucle interne manuelle (LeaveOneGroupOut + refit d'un Ridge par alpha) par
        `RidgeCV(cv=None)`, qui sélectionne l'alpha via un Leave-One-Out calculé
        analytiquement (aucun refit par échantillon ni par alpha) et qui fonctionne par TR.

        Args :
            grille_alphas : valeurs d'alpha à tester.
            n_folds : nombre de folds externes.
            test_size : part du test externe de chaque fold, exprimée dans l'unité de
                `niveau_split` (proportion si float, nombre de groupes si int).
            niveau_split : "session" ou "run", cf. `_splitter_externe`.
            n_buffer : runs écartés de part et d'autre du test si niveau_split="run"
                (0 = aucun buffer), cf. `_splitter_externe`.
            seed : graine du split externe.
            masque_roi : vecteur booléen par voxel (cf. `_charger_masque_roi`) pour
                restreindre l'analyse à une ROI ; None = cerveau entier.

        Returns :
            ResultatsCV : champs `best_alphas_inner` et Pearson à None (pas de sous-CV
                interne explicite ici, l'alpha est choisi en interne par `RidgeCV`).

        Notes :
            LOO (par timepoint) au lieu de LOGO (par session) est la seule différence
            algorithmique avec `nested_cross_validation_full_manuel`. Le LOO ignore la
            structure de session/autocorrélation temporelle que LOGO respectait pour
            la sélection d'alpha : les alphas et R² obtenus ne sont donc pas
            strictement comparables scientifiquement entre les deux méthodes,
            seulement en termes de mécanique/temps de calcul.
        """
        X, Y, groupes, TSNR = self._selection_X_Y(masque_roi=masque_roi)
        n_features = Y.shape[1]

        outer_cv, groupes_split = self._splitter_externe(niveau_split, groupes, n_folds, test_size, seed, n_buffer=n_buffer)

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)

        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y, groupes_split)):
            print(f"  -> Début du Fold externe {i + 1}/{n_folds}...")

            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

            print(f"    Train : {len(train_idx)} samples, Test : {len(test_idx)} samples")
            self._afficher_composition_runs(groupes, self._runs_par_echantillon, train_idx, test_idx)

            # alpha_per_target=True : un alpha par voxel, uniquement compatible avec cv=None (LOO).
            # TransformedTargetRegressor standardise/dé-standardise Y automatiquement.
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

        return ResultatsCV(
            r2_moyen=r2_moyen,
            r2_variance_inter_folds=r2_variance_inter_folds,
            r2_tous_les_tests=r2_tous_les_tests,
            alphas_tous_externes=alphas_tous_externes,
            alphas_tous_externes_moyen=alphas_tous_externes_moyen,
            TSNR=TSNR,
        )

    def nested_cross_validation_chunked_trimmed_ridgecv_loo(self, grille_alphas, n_folds=5, chunk_length=None, trim_size=5, seed=None, masque_roi=None):
        """Jumeau de `nested_cross_validation_ridgecv_loo` : même boucle interne
        (`RidgeCV(cv=None)`, alpha par voxel via LOO analytique), mais split externe
        remplacé par `create_chunked_folds_trimmed` (litcoder_core, voir
        `src/litcoder_folding.py`) au lieu de `GroupShuffleSplitSession`.

        Contrairement au splitter à tirage aléatoire, ce découpage est une PARTITION :
        chaque chunk est en test exactement une fois sur les `n_folds` folds.

        Args :
            grille_alphas : valeurs d'alpha à tester.
            n_folds : nombre de folds externes.
            chunk_length : taille des chunks en TR. Laisser à None (défaut) pour
                découper sur les RUNS réels via `runs=self._runs_par_echantillon` :
                aucun run n'est alors coupé entre train et test, quelle que soit sa
                longueur après alignement. Une valeur explicite rebascule sur le
                découpage positionnel de litcoder, qui n'évite la coupure que si tous
                les runs font exactement `chunk_length` TR — un seul run plus court
                décale tous les chunks suivants.
            trim_size : nombre de TR retirés aux deux bords de chaque chunk de test
                (réduction de la fuite par autocorrélation), le chunk de train
                correspondant reste entier.
            seed : graine du tirage (mélange des chunks) et du split externe.
            masque_roi : vecteur booléen par voxel (cf. `_charger_masque_roi`) pour
                restreindre l'analyse à une ROI ; None = cerveau entier.

        Returns :
            ResultatsCV : champs `best_alphas_inner` et Pearson à None.

        Notes :
            Contrairement à `GroupShuffleSplitRun`, `create_chunked_folds_trimmed` ne
            retire pas du train les chunks adjacents aux chunks de test : seul le
            trimming interne aux chunks de test protège contre l'autocorrélation.
        """
        X, Y, groupes, TSNR = self._selection_X_Y(masque_roi=masque_roi)
        n_samples = X.shape[0]
        n_features = Y.shape[1]

        rng = np.random.default_rng(seed)
        folds = create_chunked_folds_trimmed(
            n_samples, n_folds, chunk_length, trim_size=trim_size, shuffle=True, rng=rng,
            runs=self._runs_par_echantillon if chunk_length is None else None,
        )

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)

        for i, (train_idx, test_idx) in enumerate(folds):
            print(f"  -> Début du Fold externe {i + 1}/{n_folds}...")

            train_idx, test_idx = np.array(train_idx), np.array(test_idx)
            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]
            
            print(f"    Train : {len(train_idx)} samples, Test : {len(test_idx)} samples")
            self._afficher_composition_runs(groupes, self._runs_par_echantillon, train_idx, test_idx)

            # alpha_per_target=True : un alpha par voxel, uniquement compatible avec cv=None (LOO).
            # TransformedTargetRegressor standardise/dé-standardise Y automatiquement.
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

        return ResultatsCV(
            r2_moyen=r2_moyen,
            r2_variance_inter_folds=r2_variance_inter_folds,
            r2_tous_les_tests=r2_tous_les_tests,
            alphas_tous_externes=alphas_tous_externes,
            alphas_tous_externes_moyen=alphas_tous_externes_moyen,
            TSNR=TSNR,
        )

    def _generer_folds_one_cycle(self):
        """Construit les folds Train/Validation/Test/Buffer du protocole 'one cycle'
        (sessions numérotées 1..36).

        Returns :
            list[dict] : un dict par fold, clés "train"/"validation"/"test"/"buffer"
                (listes de numéros de session).

        Notes :
            - Test FIXE, identique pour tous les folds : {14, 15, 16}.
            - Buffer autour du Test, toujours exclu (jamais Train ni Validation) : {13, 17}.
            - Validation et son buffer : donnés explicitement par `FOLDS_VALIDATION_ONE_CYCLE`
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

    def nested_cross_validation_one_cycle(self, grille_alphas, masque_roi=None):
        """Validation croisée par blocs de sessions fixes : Le Test est fixe :
        sessions 14-16, jamais utilisé pour choisir les alphas et identique pour tous les folds ; 
        chaque fold ne comporte qu'un seul split Train / Validation pour sélectionner l'alpha
        optimal par voxel/parcelles, puis le modèle final est réentraîné sur Train+Validation
        et évalué une fois sur le Test fixe. Des sessions tampons ("buffer") isolent
        Test et Validation de leurs voisines pour limiter la fuite par autocorrélation
        temporelle inter-sessions. Voir `_generer_folds_one_cycle` pour le découpage.

        Args :
            grille_alphas : valeurs d'alpha à tester.
            masque_roi : vecteur booléen par voxel (cf. `_charger_masque_roi`) pour
                restreindre l'analyse à une ROI ; None = cerveau entier.

        Returns :
            ResultatsCV : seule méthode à renseigner les champs Pearson
                (`pearson_moyen`, `pearson_variance_inter_folds`,
                `pearson_tous_les_tests`) ; `best_alphas_inner` reste à None.

        Notes :
            - Un seul bloc de Validation par fold (pas plusieurs folds internes à
              agréger) : le R² de sélection est calculé en unités brutes (prédictions
              dé-standardisées), par cohérence avec les deux autres jumeaux, même si
              le R² est en réalité invariant à une transformation affine.
            - Le Pearson est une seconde lecture de la même prédiction, pas une
              seconde carte : il ignore le biais et l'échelle que le R² pénalise, donc
              il est mécaniquement plus haut. Il n'est affiché que dans la figure
              accuracy (aucune carte cérébrale dédiée).
        """
        X, Y, groupes, TSNR = self._selection_X_Y(masque_roi=masque_roi)
        n_features = Y.shape[1]

        folds = self._generer_folds_one_cycle()
        n_folds = len(folds)

        print(f"  -> {n_folds} folds 'one cycle' générés :")
        for i, fold in enumerate(folds):
            print(f"     Fold {i + 1} | Validation={fold['validation']} | Buffer={fold['buffer']} | Train={len(fold['train'])} sessions")

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        pearson_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)

        # Indices des folds réellement évalués : un fold sauté (Validation ou Test vide)
        # laisserait sinon une ligne de zéros que les agrégations ci-dessous compteraient
        # comme un vrai R² nul, tirant la moyenne vers le bas.
        folds_evalues = []

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

            r2_par_alpha = np.zeros((len(grille_alphas), n_features))
            for a_idx, alpha in enumerate(grille_alphas):
                ridge_selection = Ridge(alpha=alpha)
                ridge_selection.fit(X_train_scaled_selection, Y_train_scaled_selection)
                Y_val_pred_scaled = ridge_selection.predict(X_val_scaled)

                # Retour en unités brutes avant de scorer, comme _RidgeGCV._score()
                # qui compare toujours ses prédictions à "unscaled_y" (jamais à la
                # cible standardisée utilisée pour le fit).
                Y_val_pred_brut = scaler_Y_selection.inverse_transform(Y_val_pred_scaled)
                r2_par_alpha[a_idx, :] = r2_score(Y_val, Y_val_pred_brut, multioutput='raw_values')

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

            # `pearsonr` est vectorisé par colonne via `axis=0` (SciPy >= 1.13) : une
            # corrélation par voxel/parcelle, sans boucle Python. Un voxel plat ou une
            # prédiction plate donne NaN (écart-type nul) — ramené à 0, comme un
            # "aucune corrélation détectable", pour ne pas contaminer les agrégations.
            pearson_score_fold = np.nan_to_num(
                pearsonr(Y_test_scaled, Y_pred_scaled, axis=0).statistic, nan=0.0
            )

            r2_tous_les_tests[i, :] = r2_score_fold
            pearson_tous_les_tests[i, :] = pearson_score_fold
            folds_evalues.append(i)
            print(f"-> R2 max : {np.max(r2_score_fold)}")
            print(f"-> Pearson max : {np.max(pearson_score_fold)}")

            del ridge_final, Y_pred_scaled
            gc.collect()

        if not folds_evalues:
            raise ValueError(
                f"Aucun fold 'one cycle' évaluable pour {self.subject} : les sessions de "
                f"Test {sorted(set(SESSIONS_TEST_ONE_CYCLE))} sont absentes des données."
            )

        # On restreint AVANT d'agréger : les folds sautés ne doivent peser ni sur la
        # moyenne, ni sur la variance inter-folds, ni sur les tableaux renvoyés (qui
        # alimentent les barres d'erreur de `plot_accuracy`).
        if len(folds_evalues) < n_folds:
            print(f"  -> {n_folds - len(folds_evalues)} fold(s) sauté(s) sur {n_folds} : "
                  "exclus des agrégations et des tableaux renvoyés.")
        r2_tous_les_tests = r2_tous_les_tests[folds_evalues]
        pearson_tous_les_tests = pearson_tous_les_tests[folds_evalues]
        alphas_tous_externes = alphas_tous_externes[folds_evalues]

        r2_moyen = np.mean(r2_tous_les_tests, axis=0)
        r2_variance_inter_folds = np.var(r2_tous_les_tests, axis=0)
        pearson_moyen = np.mean(pearson_tous_les_tests, axis=0)
        pearson_variance_inter_folds = np.var(pearson_tous_les_tests, axis=0)
        alphas_tous_externes_moyen = np.mean(alphas_tous_externes, axis=0)

        return ResultatsCV(
            r2_moyen=r2_moyen,
            r2_variance_inter_folds=r2_variance_inter_folds,
            r2_tous_les_tests=r2_tous_les_tests,
            alphas_tous_externes=alphas_tous_externes,
            alphas_tous_externes_moyen=alphas_tous_externes_moyen,
            TSNR=TSNR,
            pearson_moyen=pearson_moyen,
            pearson_variance_inter_folds=pearson_variance_inter_folds,
            pearson_tous_les_tests=pearson_tous_les_tests,
        )

if __name__ == "__main__":
    # Point d'entrée : pour chaque sujet, lance les méthodes de validation croisée
    # ("one_cycle" — protocole CNeuroMod-THINGS —, et son jumeau "ridgecv_loo") sur
    # plusieurs zones cérébrales ("scopes"), différentes selon la précision :
    # - précision parcelle : toutes les parcelles / parcelles visuelles (Vis) /
    #   parcelles visuelles + attention dorsale (Vis + DorsAttn) ;
    # - précision voxel : cerveau entier / union des ROIs rétinotopiques visuelles
    #   et des ROIs catégorielles (fLoc).

    # --- PARAMÈTRES ---
    plateforme = ["Rorqual", "Mac"]
    plateforme = plateforme[1]

    liste_sujets = ["sub-01", "sub-02", "sub-03", "sub-06"]
    liste_sujets = liste_sujets[2:3]
    LAYER = "encoder_layer7_ffn"

    flag_delai_bold_brute = True
    centrage_donne_temps  = False
    flag_precision_voxel  = False   
    ROImask_flag          = False

    alphas = np.logspace(-1, 10, 20)

    for SUB in liste_sujets:
        print(f"\n{'='*60}\n  Sujet : {SUB}\n{'='*60}")

        ridge = RidgeRegression(
            plateforme, SUB, LAYER,
            flag_delai_bold_brute, centrage_donne_temps,
            flag_precision_voxel, ROImask_flag, randomize_flag=False
        )

        # Les figures sont produites par une classe séparée, qui ne calcule rien : on lui
        # donne les chemins déjà résolus et le contexte d'affichage, elle reçoit ensuite
        # les tableaux renvoyés par les méthodes de validation croisée.
        figures = VisualisationResultats(
            chemins=ridge.get_path_file_by_plateform(plateforme),
            subject=SUB,
            layer=LAYER,
            flag_precision_voxel=flag_precision_voxel,
        )

        # Zones cérébrales à analyser ("scopes"), différentes selon la précision :
        # les ROIs voxelwise (fichier ROImask) et les réseaux Yeo-7 par parcelle
        # (fichier d'annotations de l'atlas cneuromod26) ne sont pas la même chose.
        # `composition_scopes` légende la figure de comparaison : sans elle, un scope
        # nommé "ROIs" ou "visuelles" ne dit pas quelles aires il recouvre. On réutilise
        # les MÊMES constantes que les masques ci-dessus, jamais une liste recopiée à la
        # main, sinon la légende finirait par mentir sur ce qui a été calculé.
        if flag_precision_voxel:
            masque_rois = ridge._charger_masque_roi(ROIS_RETINOTOPIQUES + ROIS_CATEGORIELLES)
            scopes = {
                "cerveau_entier": None,
                "ROIs": masque_rois,
            }
            composition_scopes = {"ROIs": ROIS_RETINOTOPIQUES + ROIS_CATEGORIELLES}
        else:
            masque_visuelles = ridge._charger_masque_parcelles(RESEAUX_PARCELLES_VISUEL)
            masque_visuelles_dorsAttn = ridge._charger_masque_parcelles(RESEAUX_PARCELLES_VISUEL_DORSATTN)
            scopes = {
                "toutes_parcelles": None,
                "visuelles": masque_visuelles,
                "visuelles_dorsAttn": masque_visuelles_dorsAttn,
            }
            composition_scopes = {
                "visuelles": RESEAUX_PARCELLES_VISUEL,
                "visuelles_dorsAttn": RESEAUX_PARCELLES_VISUEL_DORSATTN,
            }

        # `seed` est fixé pour que les variantes soient comparables entre elles :
        # sans lui le split externe change à chaque exécution.
        methodes = {
            # Seule méthode qui produit aussi un score de Pearson (second panneau de la
            # figure accuracy) : elle n'a pas de `seed`, son découpage est fixé par le
            # protocole CNeuroMod-THINGS.
            "one_cycle": lambda masque: ridge.nested_cross_validation_one_cycle(alphas, masque_roi=masque),
            #"ridgecv_loo_session": lambda masque: ridge.nested_cross_validation_ridgecv_loo(alphas, n_folds=10, test_size=0.1, niveau_split="session", seed=49, masque_roi=masque),
            # test_size=0.1 -> 21 runs de test sur 213, plus 30 à 40 runs de buffer
            # (voisins immédiats), donc un train de 152-158 runs au lieu de 192.
            #"ridgecv_loo_run": lambda masque: ridge.nested_cross_validation_ridgecv_loo(alphas, n_folds=10, test_size=0.1, niveau_split="run", seed=49, masque_roi=masque),
            # Contrôle à activer si l'écart session/run est net : même split par run mais
            # SANS buffer, donc même taille de train qu'au niveau session. L'écart qui
            # subsiste alors ne vient plus du volume de données mais des runs frères de
            # la session testée restés en train (cf. docstring de GroupShuffleSplitRun).
            #"ridgecv_loo_run_sans_buffer": lambda masque: ridge.nested_cross_validation_ridgecv_loo(alphas, n_folds=10, test_size=0.1, niveau_split="run", n_buffer=0, seed=49, masque_roi=masque),
            # chunk_length=None -> chunks = runs réels (partition : chaque run testé une fois)
            #"chunked_trimmed_ridgecv_loo": lambda masque: ridge.nested_cross_validation_chunked_trimmed_ridgecv_loo(alphas, n_folds=10, trim_size=5, seed=49, masque_roi=masque),
            # Coûteux (triple boucle folds × sessions internes × alphas, pas de raccourci
            #"full_manuel": lambda masque: ridge.nested_cross_validation_full_manuel(alphas, n_folds=10, test_size=0.1, niveau_split="run", seed=49, masque_roi=masque),
        }

        # Scope sur lequel la CV est RÉELLEMENT calculée. None = tout (cerveau entier
        # ou toutes les parcelles). Chaque voxel étant régressé indépendamment des
        # autres, les scopes plus petits se déduisent ensuite par simple sélection de
        # colonnes : inutile de relancer la CV, les valeurs sont identiques au bit
        # près (cf. `ResultatsCV.restreindre` et test/test_equivalence_scopes.py).
        # Le restreindre limite mécaniquement les scopes comparables, qui doivent en
        # être des sous-ensembles.
        scope_cv = None

        # Scope qui reçoit la planche détaillée (cartes cérébrales, histogrammes
        # d'alphas, ROImask). Les autres n'apparaissent que dans la comparaison.
        scope_detaille = "cerveau_entier" if flag_precision_voxel else "toutes_parcelles"

        for nom_methode, executer in methodes.items():
            print(f"\n{'-'*60}\n[{nom_methode}] — {SUB}\n{'-'*60}")

            # UNE seule CV par méthode, donc un seul chargement des données : c'est
            # `_selection_X_Y` -> `create_X_Y_total()` qui domine le temps d'exécution.
            resultats = executer(scope_cv)

            # 1. Synthèse des scopes, sans aucune CV en plus : ce sont les mêmes
            # résultats restreints à des sous-ensembles de colonnes. Calculée d'abord,
            # car elle est passée à la planche, qui l'intègre au lieu d'un second PNG.
            r2_par_scope, pearson_par_scope = {}, {}
            for nom_scope, masque in scopes.items():
                if not scope_disponible(masque, scope_cv):
                    print(f"  -> scope {nom_scope} ignoré : hors du scope de CV.")
                    continue
                restreints = resultats.restreindre(masque_relatif(masque, scope_cv))
                r2_par_scope[nom_scope] = restreints.r2_tous_les_tests
                if restreints.pearson_tous_les_tests is not None:
                    pearson_par_scope[nom_scope] = restreints.pearson_tous_les_tests

            # 2. Planche UNIQUE : figures détaillées du scope désigné + comparaison.
            masque_detaille = scopes[scope_detaille]
            if not scope_disponible(masque_detaille, scope_cv):
                raise ValueError(
                    f"scope_detaille={scope_detaille!r} n'est pas inclus dans le scope "
                    "sur lequel la CV a tourné : impossible d'en tirer une planche."
                )
            figures.generer_toutes_les_figures(
                f"{nom_methode}_{scope_detaille}",
                resultats.restreindre(masque_relatif(masque_detaille, scope_cv)),
                alphas,
                masque_roi=masque_detaille,
                r2_par_scope=r2_par_scope,
                pearson_par_scope=pearson_par_scope or None,
                composition_scopes=composition_scopes,
            )

        print(f"\nTerminé pour le sujet {SUB}. Toutes les figures ont été sauvegardées.")
