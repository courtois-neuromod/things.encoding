from pathlib import Path
import h5py
import subprocess
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import convolve
from nilearn.glm.first_level.hemodynamic_models import spm_hrf


class TribeHDF5Normalization:
    def __init__(self, chemin_tribe, chemin_cneuromod, chemin_video,
                 tribe_ses, tribe_run, tribe_layer,
                 cneuromod_ses, cneuromod_dataset,
                 t_Tribe_s, TR_irmf_s, flag_delai_bold_brute, centrage_donne_temps):
        """
        Initialise le normalisateur avec les chemins, les clés HDF5 et les constantes de temps.
        """
        self.chemin_tribe = chemin_tribe
        self.chemin_cneuromod = chemin_cneuromod
        self.chemin_video = chemin_video

        # Clés pour naviguer dans l'HDF5 de TRIBE
        self.tribe_ses = tribe_ses
        self.tribe_run = tribe_run
        self.tribe_layer = tribe_layer

        # Clés pour naviguer dans l'HDF5 de Cneuromod
        self.cneuromod_ses = cneuromod_ses
        self.cneuromod_dataset = cneuromod_dataset

        # Constantes temporelles
        self.t_Tribe_s = t_Tribe_s
        self.TR_irmf_s = TR_irmf_s

        # Variables qui stockeront les matrices finales prêtes pour la Ridge
        self.X_aligne = None
        self.Y_cible = None

        self.flag_delai_bold_brute = flag_delai_bold_brute
        self.centrage_donne_temps = centrage_donne_temps

    def _obtenir_duree_video(self):
        """Méthode privée pour extraire la durée via ffprobe."""
        commande_metadata_video = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1",
            str(self.chemin_video)
        ]
        resultat = subprocess.run(commande_metadata_video, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        duree_reelle = float(resultat.stdout.strip())
        print(f"[Info] Durée lue par ffprobe : {duree_reelle} secondes")
        return duree_reelle

    def executer_pipeline(self):
        """Exécute l'intégralité du pipeline de nettoyage et d'alignement."""
        print("[Traitement en cours] Ouverture des fichiers HDF5...")

        with h5py.File(self.chemin_tribe, 'r') as tribe_hdf5, \
                h5py.File(self.chemin_cneuromod, 'r') as cneuromod_hdf5:
            # 1. Extraction Cneuromod (Cible Y) avec clés dynamiques
            dataset_cneuromod = cneuromod_hdf5[self.cneuromod_ses][self.cneuromod_dataset][:]
            self.Y_cible = dataset_cneuromod
            print(f"Dataset Cneuromod (Y) : {self.Y_cible.shape}")

            # 2. Extraction et Aplatissement Tribe
            dataset_tribe = tribe_hdf5[self.tribe_ses][self.tribe_run][self.tribe_layer]
            print(f"Dataset Tribe origin : {dataset_tribe.shape}")

            dataset_tribe_concatene = dataset_tribe[:].reshape(-1, 1152)
            print(f"Dataset Tribe concatene : {dataset_tribe_concatene.shape}")

            # 3. Nettoyage Temporel (Coup de ciseaux)
            duree_reelle = self._obtenir_duree_video()
            nb_instants_valides = int(duree_reelle / self.t_Tribe_s)
            dataset_tribe_bonne_duree = dataset_tribe_concatene[:nb_instants_valides, :]
            print(f"Dataset Tribe nettoyé du padding : {dataset_tribe_bonne_duree.shape}")

            if self.flag_delai_bold_brute == True:
                if self.centrage_donne_temps == True:
                    temps_source = np.arange(dataset_tribe_bonne_duree.shape[0]) * self.t_Tribe_s + self.t_Tribe_s / 2
                    temps_cible = np.arange(self.Y_cible.shape[0]) * self.TR_irmf_s + self.TR_irmf_s / 2
                    temps_cible_avec_delai_bold = temps_cible - 5
                else:
                    temps_source = np.arange(dataset_tribe_bonne_duree.shape[0]) * self.t_Tribe_s
                    temps_cible = np.arange(self.Y_cible.shape[0]) * self.TR_irmf_s
                    temps_cible = temps_cible - 5

                dataset_tribe_propre = dataset_tribe_bonne_duree
            else:
                #Création d'un signal hemodynamique de 30s
                noyau_hrf = spm_hrf(t_r=self.t_Tribe_s)
                # Reshape pour avoir le même nombre de dimensions que dataset_tribe
                noyau_hrf = noyau_hrf.reshape(-1, 1)

                dataset_tribe_propre = convolve(dataset_tribe_bonne_duree, noyau_hrf, mode='full')
                dataset_tribe_propre = dataset_tribe_propre[:nb_instants_valides, :]

                # 4. Création des axes temporels
                temps_source = np.arange(dataset_tribe_propre.shape[0]) * self.t_Tribe_s + self.t_Tribe_s/2
                temps_cible = np.arange(self.Y_cible.shape[0]) * self.TR_irmf_s + self.TR_irmf_s/2

            masque_debut = temps_cible >= temps_source[0]
            masque_fin = temps_cible <= temps_source[-1]
            masque_valide = masque_debut & masque_fin

            # 5. Interpolation (L'alignement)
            aligneur_temporel = interp1d(temps_source, dataset_tribe_propre, axis=0, bounds_error=False, fill_value=0.0)
            self.X_aligne = aligneur_temporel(temps_cible)

            self.X_aligne = self.X_aligne[masque_valide]
            self.Y_cible = self.Y_cible[masque_valide]

            print(f"X (Tribe) : {self.X_aligne.shape} == Y (Cneuromod) : {self.Y_cible.shape}")

        return self.X_aligne, self.Y_cible


if __name__ == "__main__":
    # 1. Définition des chemins dynamiques
    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data"
    chemin_video = DATA_DIR / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1.mp4"
    chemin_tribe = ROOT / "output" / "hdf5" / "sub-01.h5"
    chemin_cneuromod = ROOT / "data" / "timeseries" / "cneuromod2026" / "sub-01" / "sub-01_task-things_space-MNI152NLin2009cAsym_atlas-cneuromod26_desc-1134Parcels_timeseries.h5"

    # 2. Instanciation (ici tu peux changer n'importe quelle clé HDF5 !)
    normalisateur = TribeHDF5Normalization(
        chemin_tribe=chemin_tribe,
        chemin_cneuromod=chemin_cneuromod,
        chemin_video=chemin_video,
        tribe_ses='ses-001',
        tribe_run='run-1',
        tribe_layer='encoder_layer7_ffn',
        cneuromod_ses='ses-01',
        cneuromod_dataset='ses-01_task-things_run-1_timeseries',
        t_Tribe_s=0.5,
        TR_irmf_s=1.49,
        flag_delai_bold_brute = True,
        centrage_donne_temps = False,
    )

    # 3. Exécution du traitement
    X, Y = normalisateur.executer_pipeline()
