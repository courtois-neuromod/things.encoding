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

from collections import defaultdict
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

    nb_stimuli = 0

    fmri_enc = model.__pydantic_private__['_model']
    #for file in Path(STIMULI_DIR).rglob("*.mp4"):
    for i in range(1):
        file = "stimulus_0000.mp4"
        events = model.get_events_dataframe(video_path=str(STIMULI_DIR / file))

        features = defaultdict(list)
        _hooks = []

        def _make_hook(name: str):
            def hook(module, input, output):
                out = output[0] if isinstance(output, tuple) else output
                features[name].append(out.detach().cpu())

            return hook

        # Attacher
        features.clear()
        _hooks.clear()

        for i, layer_block in enumerate(fmri_enc.encoder.layers):
            if i % 2 == 0:  # attention only
                name = f'encoder.layer{i // 2}.attn'
                _hooks.append(layer_block[1].register_forward_hook(_make_hook(name)))

        print(f"Hooks enregistrés : {len(_hooks)}")
        print(events)

        """"
        # --- forward pass ---
        
        model.predict(events)

        # Récupérer les features
        result = {}
        for layer_name, tensors in features.items():
            stacked = torch.cat(tensors, dim=0)
            result[layer_name.replace('.', '_')] = stacked.numpy()
            print(f"  {layer_name:40s}  shape={tuple(stacked.shape)}")

        # Retirer
        for h in _hooks:
            h.remove()
        _hooks.clear()

    # Extraction des représentations latentes via forward hooks
    fmri_enc = model.__pydantic_private__['_model']
    hooks = TransformerHooks(fmri_enc)
    hooks.attacher()

    events = model.get_events_dataframe(video_path=str(stimulus_path))
    import torch
    with torch.no_grad():
        preds, segments = model.predict(events)

    hooks.retirer()
    features = hooks.get_features()

    print(f"preds shape: {preds.shape}")

    # Sauvegarde HDF5
    writer = HDF5Writer(HDF5_PATH)
    writer.sauvegarder(features, subject, session, run, nb_stimuli)
    """