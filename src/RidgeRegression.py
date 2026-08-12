"""Régression Ridge pour l'encodage cérébral THINGS memory.

Entraîne une Ridge (grille d'alphas balayée manuellement) par couche et
évalue la prédiction via trois variantes de validation croisée imbriquée.
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
from sklearn.model_selection import LeaveOneGroupOut, LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from GroupShuffleSplitSession import GroupShuffleSplitSession
from litcoder_folding import create_chunked_folds_trimmed
from TribeHDF5Normalization import TribeHDF5Normalization

# Ignore spécifiquement les avertissements de matrices mal conditionnées
warnings.filterwarnings(action='ignore', category=LinAlgWarning)

# Idem pour les débordements numpy (divide/overflow/invalid) rencontrés lors des
# fits Ridge avec les plus grands alphas de la grille (jusqu'à 1e10) : ils produisent
# potentiellement des coefficients NaN/Inf pour CES alphas précis, mais n'empêchent
# pas la sélection de l'alpha optimal (leur R² correspondant devient alors très
# négatif ou NaN, donc jamais retenu par argmax).
np.seterr(divide='ignore', over='ignore', invalid='ignore')

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
            ROImask_flag : non utilisé pour l'instant (réservé).
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
        else:
            ROOT_ENCODING = Path(__file__).parent.parent
            ROOT_TIMESERIES = ROOT_ENCODING / "data"

        chemin_tribe = ROOT_ENCODING / "output" / "features" / "things_encoding" / f"{self.subject}.h5"
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

    def nested_cross_validation_full_manuel(self, grille_alphas, n_folds=5, test_size=0.2, seed=None, masque_roi=None):
        """Validation croisée imbriquée 100% manuelle : sélection d'alpha par voxel/
        parcelle via `LeaveOneGroupOut` (une session isolée à la fois) et refit
        explicite d'un `Ridge` pour chaque alpha de la grille sur chaque fold interne.

        Args :
            grille_alphas : valeurs d'alpha à tester.
            n_folds : nombre de folds externes (GroupShuffleSplitSession).
            test_size : proportion de sessions dans le test externe de chaque fold.
            seed : graine du split externe.
            masque_roi : vecteur booléen par voxel (cf. `_charger_masque_roi`) pour
                restreindre l'analyse à une ROI ; None = cerveau entier.

        Returns :
            tuple : (r2_moyen, r2_variance_inter_folds, r2_tous_les_tests,
                alphas_tous_externes, alphas_tous_externes_moyen, best_alphas_inner,
                TSNR).

        Notes :
            - Jumeau de `nested_cross_validation_ridgecv_loo` : même split externe et
              même principe (un alpha par voxel), mais boucle interne manuelle (LOGO)
              plutôt que LOO analytique 
            - La sélection d'alpha utilise un R² poolé (résidus accumulés sur tous les
              folds internes puis un seul calcul de R², jamais une moyenne de R² par
              fold), pour reproduire exactement le mécanisme de `RidgeCV(cv=None)`.
        """
        X, Y, groupes, TSNR = self._selection_X_Y(masque_roi=masque_roi)
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

        return r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, best_alphas_inner, TSNR

    def nested_cross_validation_ridgecv_loo(self, grille_alphas, n_folds=5, test_size=0.2, seed=None, masque_roi=None):
        """Jumeau sklearn-natif de `nested_cross_validation_full_manuel` : remplace la
        boucle interne manuelle (LeaveOneGroupOut + refit d'un Ridge par alpha) par
        `RidgeCV(cv=None)`, qui sélectionne l'alpha via un Leave-One-Out calculé
        analytiquement (aucun refit par échantillon ni par alpha) et qui fonctionne par TR.

        Args :
            grille_alphas : valeurs d'alpha à tester.
            n_folds : nombre de folds externes (GroupShuffleSplitSession).
            test_size : proportion de sessions dans le test externe de chaque fold.
            seed : graine du split externe.
            masque_roi : vecteur booléen par voxel (cf. `_charger_masque_roi`) pour
                restreindre l'analyse à une ROI ; None = cerveau entier.

        Returns :
            tuple : (r2_moyen, r2_variance_inter_folds, r2_tous_les_tests,
                alphas_tous_externes, alphas_tous_externes_moyen, TSNR).

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

        outer_cv = GroupShuffleSplitSession(n_splits=n_folds, test_size=test_size, random_state=seed)

        r2_tous_les_tests = np.zeros((n_folds, n_features), dtype=np.float32)
        alphas_tous_externes = np.zeros((n_folds, n_features), dtype=np.float64)

        for i, (train_idx, test_idx) in enumerate(outer_cv.split(X, Y, groupes)):
            print(f"  -> Début du Fold externe {i + 1}/{n_folds}...")

            X_train, Y_train = X[train_idx], Y[train_idx]
            X_test, Y_test = X[test_idx], Y[test_idx]

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

        return r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, TSNR

    def nested_cross_validation_chunked_trimmed_ridgecv_loo(self, grille_alphas, n_folds=5, chunk_length=190, trim_size=5, seed=None, masque_roi=None):
        """Jumeau de `nested_cross_validation_ridgecv_loo` : même boucle interne
        (`RidgeCV(cv=None)`, alpha par voxel via LOO analytique), mais split externe
        remplacé par `create_chunked_folds_trimmed` (litcoder_core, voir
        `src/litcoder_folding.py`) au lieu de `GroupShuffleSplitSession`.

        Args :
            grille_alphas : valeurs d'alpha à tester.
            n_folds : nombre de folds externes.
            chunk_length : taille des chunks (en TR) découpés dans X/Y avant tirage.
                DOIT être un multiple de la longueur d'un run (190 TR pour things,
                cf. test/test_litcoder_folding.py) sous peine de couper des runs
                entre train et test (fuite temporelle)
            trim_size : nombre de TR retirés aux deux bords de chaque chunk de test
                (réduction de la fuite par autocorrélation), le chunk de train
                correspondant reste entier.
            seed : graine du tirage (mélange des chunks) et du split externe.
            masque_roi : vecteur booléen par voxel (cf. `_charger_masque_roi`) pour
                restreindre l'analyse à une ROI ; None = cerveau entier.

        Returns :
            tuple : (r2_moyen, r2_variance_inter_folds, r2_tous_les_tests,
                alphas_tous_externes, alphas_tous_externes_moyen, TSNR).

        Notes :
            Contrairement à `GroupShuffleSplitSession`, `create_chunked_folds_trimmed`
            ne retire pas du train les chunks adjacents aux chunks de test : seul le
            trimming interne aux chunks de test protège contre l'autocorrélation.
        """
        X, Y, groupes, TSNR = self._selection_X_Y(masque_roi=masque_roi)
        n_samples = X.shape[0]
        n_features = Y.shape[1]

        rng = np.random.default_rng(seed)
        folds = create_chunked_folds_trimmed(
            n_samples, n_folds, chunk_length, trim_size=trim_size, shuffle=True, rng=rng
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
            """
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

        return r2_moyen, r2_variance_inter_folds, r2_tous_les_tests, alphas_tous_externes, alphas_tous_externes_moyen, TSNR
        """
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
            tuple : (r2_moyen, r2_variance_inter_folds, r2_tous_les_tests,
                alphas_tous_externes, alphas_tous_externes_moyen, TSNR).

        Notes :
            Un seul bloc de Validation par fold (pas plusieurs folds internes à
            agréger) : le R² de sélection est calculé en unités brutes (prédictions
            dé-standardisées), par cohérence avec les deux autres jumeaux, même si
            le R² est en réalité invariant à une transformation affine.
        """
        X, Y, groupes, TSNR = self._selection_X_Y(masque_roi=masque_roi)
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
        chemins = self.get_path_file_by_plateform(self.plateforme)
        chemin_sortie = chemins.root_encoding / "output" / "analysis" / nom_fichier
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

    def plot_ROImask_histogram(self, scores_finaux):
        """Trace un barplot du R² moyen par ROI (voxelwise uniquement) et l'enregistre
        en PNG.

        Args :
            scores_finaux : R² moyen par voxel (cerveau entier, même longueur que les
                masques du fichier ROImask).

        Returns :
            Path | None : chemin du PNG sauvegardé, ou None si `flag_precision_voxel`
                est False (analyse indisponible en précision parcelle).
        """
        chemins = self.get_path_file_by_plateform(self.plateforme)
        fichier_ROImask = chemins.chemin_ROImask

        # Chaque ROI est rattachée à une famille, colorée via `palette` ci-dessous.
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

    def plot_accuracy(self, r2_tous_les_tests, suffix=""):
        """Trace les barres mean/median/top-10% (moyenne ± écart-type inter-folds)
        pour UN sujet et les enregistre en PNG. Pas d'agrégation multi-sujets : un
        seul appel = un seul sujet, une seule figure.

        Args :
            r2_tous_les_tests : R² par fold externe et par voxel/parcelle, shape
                (n_folds, n_features).
            suffix : suffixe ajouté au nom du fichier de sortie.

        Returns :
            Path : chemin du fichier PNG sauvegardé.
        """
        n_features = r2_tous_les_tests.shape[1]
        n_folds = r2_tous_les_tests.shape[0]

        # Une valeur par fold externe (pas juste par métrique) : nécessaire pour que
        # errorbar="sd" dans sns.barplot trace l'écart-type inter-folds.
        means_par_fold = np.mean(r2_tous_les_tests, axis=1)
        medians_par_fold = np.median(r2_tous_les_tests, axis=1)
        seuils_top10_par_fold = np.percentile(r2_tous_les_tests, 90, axis=1)
        top10_par_fold = np.array([
            np.mean(fold[fold >= seuil])
            for fold, seuil in zip(r2_tous_les_tests, seuils_top10_par_fold)
        ])

        df = pd.DataFrame({
            "Métrique": ["mean"] * n_folds + ["median"] * n_folds + ["top-10% mean"] * n_folds,
            "R2": np.concatenate([means_par_fold, medians_par_fold, top10_par_fold]),
        })

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.grid(True, axis='both', linestyle='-', alpha=0.2, color='grey')
        ax.set_axisbelow(True)

        palette = {"mean": "#0072B2", "median": "#56B4E9", "top-10% mean": "#E69F00"}
        sns.barplot(
            data=df, x="Métrique", y="R2", hue="Métrique",
            palette=palette, ax=ax, errorbar="sd", capsize=0.1, legend=False,
        )

        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xlabel("")
        ax.set_ylabel("R² (raw)", fontsize=12)
        ax.set_title(f"Accuracy — {self.subject} / {self.layer} (n={n_features:,}, {n_folds} folds)", fontsize=14, fontweight='bold')

        plt.tight_layout()

        nom_fichier = f"accuracy_{self.subject}_{self.layer}{suffix}.png"
        chemin_sauvegarde = self._sauvegarder_figure(fig, nom_fichier, "Accuracy sauvegardée")
        plt.close(fig)
        return chemin_sauvegarde


    def plot_alphas_histogram(self, alphas_fold, grille_alphas, alphas_finaux=None, suffix=""):
        """Trace la distribution (log10) des alphas sélectionnés et l'enregistre en PNG.

        Args :
            alphas_fold : alphas par fold externe et par voxel/parcelle (ignoré si
                `alphas_finaux` est fourni).
            grille_alphas : grille complète d'alphas testés (fixe les bins/ticks).
            alphas_finaux : si fourni, affiche la distribution des alphas moyens (une
                courbe) plutôt que la distribution empilée par fold.
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
            titre = "Distribution des alphas moyens"
        else:
            alphas_fold = np.array(alphas_fold)
            rows = [{"log10_alpha": np.log10(v), "fold": f"fold_{i + 1}"}
                    for i, fold in enumerate(alphas_fold) for v in fold]
            df = pd.DataFrame(rows)
            log10_valeurs = np.log10(alphas_fold.flatten())
            hue_params = {"hue": "fold", "multiple": "dodge", "palette": "tab20"}
            titre = "Distribution des alphas par fold"

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
        ax.set_title(titre)
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

    def regrouper_figures_dans_une_planche(self, nom_methode, liste_chemins_figures, nombre_de_colonnes=3):
        """Assemble toutes les figures PNG déjà sauvegardées pour UNE méthode de
        validation croisée dans une seule image PNG (une "planche"), au lieu d'avoir
        un fichier séparé par figure.

        Args :
            nom_methode : nom de la méthode (utilisé dans le titre et le nom de fichier).
            liste_chemins_figures : chemins renvoyés par les appels à `brain_mapping_r2`,
                `brain_mapping_alphas`, `plot_accuracy`, etc. Les entrées None (ex.
                ROImask quand `flag_precision_voxel` est False) sont ignorées.
            nombre_de_colonnes : largeur de la grille d'assemblage.

        Returns :
            Path | None : chemin du fichier PNG sauvegardé, ou None si
                `liste_chemins_figures` ne contient aucune figure valide.
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

    def generer_toutes_les_figures(self, nom_methode, resultats, grille_alphas, noms_parcelles=None, masque_roi=None):
        """Génère et sauvegarde l'ensemble des figures standard pour UNE méthode de
        validation croisée (appelée une fois par méthode : full_manuel, ridgecv_loo,
        one_cycle...).

        Args :
            nom_methode : nom de la méthode (utilisé dans les noms de fichiers).
            resultats : tuple renvoyé par la méthode `nested_cross_validation_*`
                correspondante — 7 éléments pour `nested_cross_validation_full_manuel`
                (avec `best_alphas_inner`, diagnostic de sous-CV interne LOGO absent
                des deux autres méthodes), 6 éléments sinon.
            grille_alphas : grille d'alphas testée (transmise aux histogrammes).
            noms_parcelles : transmis à `brain_mapping_r2`/`print_scores`.
            masque_roi : vecteur booléen par voxel ; si fourni, les cartes cérébrales
                projettent les valeurs sur le cerveau entier (voxels hors ROI
                masqués), et le ROImask histogram (déjà une ventilation par ROI) est
                sauté puisque l'analyse est déjà restreinte à une ROI.

        Returns :
            dict : résumé ("r2_moyen", "r2_tous_les_tests", "alphas_moyens")
                réutilisable pour comparer les méthodes entre elles dans `__main__`.
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

        # ROIMask (uniquement en précision voxel, et uniquement pour l'analyse cerveau
        # entier : sur une analyse déjà restreinte à une ROI, la ventilation par ROI
        # n'apporte rien de plus)
        if self.flag_precision_voxel and masque_roi is None:
            print(" -> ROIMask...")
            chemin_figure = self.plot_ROImask_histogram(r2_moyen)
            liste_chemins_figures.append(chemin_figure)
        elif masque_roi is None:
            print(f"  -> ROIMask ignoré pour {nom_methode} (nécessite flag_precision_voxel=True).")

        # Regroupement de toutes les figures ci-dessus dans un seul fichier PNG.
        print(" -> Assemblage de la planche de figures...")
        chemin_planche = self.regrouper_figures_dans_une_planche(nom_methode, liste_chemins_figures)

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
        }

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

        # Zones cérébrales à analyser ("scopes"), différentes selon la précision :
        # les ROIs voxelwise (fichier ROImask) et les réseaux Yeo-7 par parcelle
        # (fichier d'annotations de l'atlas cneuromod26) ne sont pas la même chose.
        if flag_precision_voxel:
            masque_rois = ridge._charger_masque_roi(ROIS_RETINOTOPIQUES + ROIS_CATEGORIELLES)
            scopes = {
                "cerveau_entier": None,
                "ROIs": masque_rois,
            }
        else:
            masque_visuelles = ridge._charger_masque_parcelles(RESEAUX_PARCELLES_VISUEL)
            masque_visuelles_dorsAttn = ridge._charger_masque_parcelles(RESEAUX_PARCELLES_VISUEL_DORSATTN)
            scopes = {
                "toutes_parcelles": None,
                "visuelles": masque_visuelles,
                "visuelles_dorsAttn": masque_visuelles_dorsAttn,
            }

        methodes = {
            #"one_cycle": lambda masque: ridge.nested_cross_validation_one_cycle(alphas, masque_roi=masque),
            "ridgecv_loo": lambda masque: ridge.nested_cross_validation_ridgecv_loo(alphas, n_folds=10, test_size=0.1, masque_roi=masque),
            #"chunked_trimmed_ridgecv_loo": lambda masque: ridge.nested_cross_validation_chunked_trimmed_ridgecv_loo(alphas, n_folds=10, chunk_length=180, seed=49, masque_roi=masque)
            # Coûteux (triple boucle folds × sessions internes × alphas, pas de raccourci
            #"full_manuel": lambda masque: ridge.nested_cross_validation_full_manuel(alphas, n_folds=10, test_size=0.1, masque_roi=masque),
        }

        for nom_scope, masque_roi in scopes.items():
            for nom_methode, executer in methodes.items():
                nom_complet = f"{nom_methode}_{nom_scope}"
                print(f"\n{'-'*60}\n[{nom_complet}] — {SUB}\n{'-'*60}")
                resultats = executer(masque_roi)
                ridge.generer_toutes_les_figures(nom_complet, resultats, alphas, masque_roi=masque_roi)

        print(f"\nTerminé pour le sujet {SUB}. Toutes les figures ont été sauvegardées.")
