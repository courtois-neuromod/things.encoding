"""
Extraction des représentations latentes de TRIBE v2 sur un stimulus unique.
Découpe la sous-vidéo correspondant au premier stimulus du run-1 (sub-01, ses-001),
la passe à TribeModel avec forward hooks, et sauvegarde les activations en HDF5.
"""
import warnings
import logging
from pathlib import Path
import pandas as pd
from torchcodec.decoders import VideoDecoder
import torch

from Config import Config
from VideoSegmenteur import VideoSegmenteur
from HDF5Writer import HDF5Writer
from TransformerHooks import TransformerHooks

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    setup = Config(
        plateforme="macos",
    )
    setup.charger_env()

    ROOT = Path(__file__).parent.parent

    TSV_PATH = ROOT / "data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1_events.tsv"
    VIDEO_PATH = ROOT / "data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1.mp4"
    VIDEO_PATH_STR = str(VIDEO_PATH)
    STIMULI_DIR = ROOT / "output/stimuli"
    HDF5_PATH = ROOT / "output/hdf5/latents.h5"

    video_path = str(VIDEO_PATH)
    df = pd.read_csv(TSV_PATH, sep="\t")

    STIMULI_DIR.mkdir(exist_ok=True)

    nb_stimuli = 0
    stimulus_path = STIMULI_DIR / f"stimulus_{nb_stimuli:04d}.mp4"

    segmenteur = VideoSegmenteur(VIDEO_PATH_STR, Path(STIMULI_DIR))

    video = VideoDecoder(video_path)
    metadata = video.metadata
    all_frames = video.get_frames_played_in_range(0, metadata.duration_seconds)
    timestamps = all_frames.pts_seconds  # tensor des timestamps en secondes
    n_frames = len(timestamps)

    segmenteur.cut_run(timestamps, n_frames)

    subject = VIDEO_PATH.parts[-3]  # "sub-01"
    session = VIDEO_PATH.parts[-2]  # "ses-001"
    run     = VIDEO_PATH.stem       # "sub-01_ses-001_task-thingsmemory_run-1"
    model   = setup.charger_modele()

    fmri_enc = model.__pydantic_private__['_model']

    nb_stimuli = 0
    for file in sorted(Path(STIMULI_DIR).glob("*.mp4")):
        stimulus_idx = int(file.stem.split("_")[-1])

        events = model.get_events_dataframe(video_path=str(file))

        hooks = TransformerHooks(fmri_enc)
        hooks.attacher()

        with torch.no_grad():
            preds, segments = model.predict(events)

        hooks.retirer()
        features = hooks.get_features()

        print(f"Stimulus {stimulus_idx} - preds shape: {preds.shape}")

        writer = HDF5Writer(HDF5_PATH)
        writer.sauvegarder(features, subject, session, run, stimulus_idx)
