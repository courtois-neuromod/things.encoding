"""
Sauvegarde des représentations latentes extraites de TRIBE v2 en HDF5.
Structure : sujet / session / run / stimulus_N / couche
"""
from pathlib import Path
import h5py


class HDF5Writer:
    """Sauvegarde les activations des couches TRIBE v2 dans un fichier HDF5.

    Structure du fichier :
        subject/session/run/stimulus_N/encoder_layerX_attn -> array numpy

    Usage :
        writer = HDF5Writer("output/latents.h5")
        writer.sauvegarder(features, subject, session, run, stimulus_idx)
    """

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    def sauvegarder(
        self,
        features: dict,
        subject: str,
        session: str,
        run: str,
        stimulus_idx: int,
    ) -> None:
        """Écrit les features dans le fichier HDF5 en mode append.

        Args:
            features: dict {safe_name: array numpy} retourné par TransformerHooks.get_features()
            subject: identifiant du sujet, ex. "sub-01"
            session: identifiant de la session, ex. "ses-001"
            run: nom du run, ex. "sub-01_ses-001_task-thingsmemory_run-1"
            stimulus_idx: index du stimulus dans le run
        """
        with h5py.File(self.output_path, 'a') as hf:
            group = hf.require_group(
                f"{subject}/{session}/{run}/stimulus_{stimulus_idx}"
            )
            for safe_name, array in features.items():
                if safe_name in group:
                    del group[safe_name]
                group.create_dataset(safe_name, data=array)

        print(f"Sauvegardé : {subject}/{session}/{run}/stimulus_{stimulus_idx} → {self.output_path}")
