"""
Sauvegarde des représentations latentes extraites de TRIBE v2 en HDF5.
Structure : sujet / session / run / couche
"""
from pathlib import Path
import h5py
import numpy as np


class HDF5Writer:
    """Sauvegarde les activations des couches TRIBE v2 dans des fichiers HDF5 par sujet.

    Structure du fichier {subject}.h5 :
        session/run/preds                -> array numpy (n_timesteps, n_vertices)
        session/run/encoder_layerX_attn  -> array numpy (n_windows, T, H)
        session/run/encoder_layerX_ffn   -> array numpy (n_windows, T, H)

    Usage :
        writer = HDF5Writer("output/latents")
        writer.sauvegarder(features, preds, subject, session, run)
    """

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def sauvegarder(
        self,
        features: dict,
        preds: np.ndarray,
        subject: str,
        session: str,
        run: str,
    ) -> None:
        """Écrit les features et preds dans le fichier HDF5 du sujet en mode append.

        Args:
            features: dict {safe_name: array numpy} retourné par TransformerHooks.get_features()
            preds: array numpy (n_timesteps, n_vertices) retourné par model.predict()
            subject: identifiant du sujet, ex. "sub-01"
            session: identifiant de la session, ex. "ses-001"
            run: nom du run, ex. "sub-01_ses-001_task-thingsmemory_run-1"
        """
        output_path = self.output_dir / f"{subject}.h5"

        with h5py.File(output_path, 'a') as hf:
            group = hf.require_group(f"{session}/{run}")

            # Sauvegarde des prédictions BOLD
            if 'preds' in group:
                del group['preds']
            group.create_dataset('preds', data=preds)

            # Sauvegarde des latents par couche
            for safe_name, array in features.items():
                if safe_name in group:
                    del group[safe_name]
                group.create_dataset(safe_name, data=array)

        print(f"Sauvegardé : {session}/{run} → {output_path}")
