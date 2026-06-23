"""
Extraction des représentations latentes de TRIBE v2 sur toutes les vidéos.
Parcourt récursivement le dossier data, passe chaque vidéo à TribeModel avec
forward hooks, et sauvegarde les activations en HDF5.
"""
import warnings
import logging
from pathlib import Path
import torch
import gc

from Config import Config
from HDF5Writer import HDF5Writer
from TransformerHooks import TransformerHooks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extraction des latents TRIBE v2')
    parser.add_argument('--subject', type=str, required=True, help='Identifiant du sujet à traiter')
    args = parser.parse_args()

    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    config = Config(
        plateforme=None,
    )
    config.charger_env()

    ROOT = Path(__file__).parent.parent
    DATA_DIR = ROOT / "data" / "things_mp4_cfr"
    HDF5_DIR = ROOT / "output/hdf5/things_encoding"

    model = config.charger_modele()
    fmri_enc = model.__pydantic_private__['_model']

    writer = HDF5Writer(HDF5_DIR)
    video_files = sorted(DATA_DIR.rglob("{args.subject}_*_task-*_desc-CFR.mp4"))
    print(f"SUjet : {arg.subject} --> {len(video_files)} vidéos trouvées")

    for video_path in video_files:
        parts = video_path.stem.split("_")
        subject = parts[0]  # "sub-01"
        session = parts[1]  # "ses-001"
        run = next((p for p in parts if p.startswith("run-")), "run-unknown")

        if not subject.startswith("sub-") or not session.startswith("ses-"):
            print(f"✗ Structure invalide : {video_path.name}",flush=True)
            continue

        outpout_path = HD5_DIR / f"{subject}.h5"
        if outpout_path.exists():
            with

        try:
            events = model.get_events_dataframe(video_path=str(video_path))

            hooks = TransformerHooks(fmri_enc)
            hooks.attacher()

            with torch.no_grad():
                preds, segments = model.predict(events)

            hooks.retirer()
            features = hooks.get_features()

            writer.sauvegarder(features, preds, subject, session, run)

            print(f"✓ {subject}/{session}/{run} - preds shape: {preds.shape}",flush=True)

            # Nettoyage de la mémoire pour la vidéo suivante
            del features, preds, segments, events, hooks

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"✗ {subject}/{session}/{run} - Erreur: {e}",flush=True)


