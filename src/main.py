"""
Extraction des représentations latentes de TRIBE v2 sur toutes les vidéos.
Parcourt récursivement le dossier data, passe chaque vidéo à TribeModel avec
forward hooks, et sauvegarde les activations en HDF5.
"""
import warnings
import logging
from pathlib import Path
import torch

from Config import Config
from HDF5Writer import HDF5Writer
from TransformerHooks import TransformerHooks

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    config = Config(
        plateforme=None,
    )
    config.charger_env()

    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "things-cfr"
    HDF5_DIR = ROOT / "output/hdf5/things_encoding"

    model = config.charger_modele()
    fmri_enc = model.__pydantic_private__['_model']

    writer = HDF5Writer(HDF5_DIR)
    video_files = sorted(DATA_DIR.glob("sub-*_*_task-*.mp4"))

    for video_path in video_files:
        parts = video_path.stem.split("_")
        subject = parts[0]
        session = parts[1]
        run = video_path.stem

        if not subject.startswith("sub-") or not session.startswith("ses-"):
            print(f"✗ Structure invalide : {video_path.name}")
            continue

        try:
            events = model.get_events_dataframe(video_path=str(video_path))

            hooks = TransformerHooks(fmri_enc)
            hooks.attacher()

            with torch.no_grad():
                preds, segments = model.predict(events)

            hooks.retirer()
            features = hooks.get_features()

            writer.sauvegarder(features, subject, session, run)

            print(f"✓ {subject}/{session}/{run} - preds shape: {preds.shape}")
        except Exception as e:
            print(f"✗ {subject}/{session}/{run} - Erreur: {e}")

