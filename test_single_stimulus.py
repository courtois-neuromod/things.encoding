"""
Extraction des représentations latentes de TRIBE v2 sur un stimulus unique.
Découpe la sous-vidéo correspondant au premier stimulus du run-1 (sub-01, ses-001),
la passe à TribeModel et récupère les prédictions fMRI.
Étape préliminaire avant l'ajout des forward hooks et la sauvegarde HDF5.
"""
import os
from pathlib import Path
import pandas as pd
from setup import Setup
import numpy as np
import matplotlib.pyplot as plt
import torch
import ffmpeg
from collections import defaultdict
import h5py

if __name__ == '__main__':
    setup = Setup(
        tsv_path=Path("data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1_events.tsv"),
        video_path="data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1.mp4",
        plateforme="macos"
    )
    setup.charger_env()
    setup.charger_modele()

    tsv_path = setup.tsv_path
    video_path = setup.video_path
    subject = Path(video_path).parts[-3]  # "sub-01"
    session = Path(video_path).parts[-2]  # "ses-001"
    run = Path(video_path).stem
    model = setup.model

    df = pd.read_csv(tsv_path, sep="\t")
    image_path = df["image_path"]

    #for i, onset in enumerate(df["onset"]):
    #print(list(enumerate(df["onset"]))[:3])
    onset = df["onset"].iloc[0]
    nb_stimuli = 0

    ffmpeg.input(video_path, ss=onset, t=2.98).output(f"tmp_stimulus_{nb_stimuli}.mp4", vcodec='hevc_videotoolbox').run()

    fmri_enc = model.__pydantic_private__['_model']
    fmri_enc.eval()

    features = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            out = output[0] if isinstance(output, tuple) else output
            features[name].append(out.detach().cpu())

        return hook

    if not Path(f"tmp_stimulus_{nb_stimuli}.mp4").exists() or Path(f"tmp_stimulus_{nb_stimuli}.mp4").stat().st_size == 0:
        print("ERREUR : fichier vidéo vide ou absent")
    else:
        # Each transformer layer: even indices = Attention, odd = FeedForward
        # encoder.layers[i][1] is the actual module (index [0] is the norm, [2] is Residual)
        for i, layer_block in enumerate(fmri_enc.encoder.layers):
            submodule = layer_block[1]  # Attention or FeedForward
            layer_type = 'attn' if i % 2 == 0 else 'ffn'
            transformer_layer_idx = i // 2  # 0-7
            if layer_type == 'attn':
                name = f'encoder.layer{transformer_layer_idx}.{layer_type}'
                hooks.append(submodule.register_forward_hook(make_hook(name)))

        events = model.get_events_dataframe(video_path=f"tmp_stimulus_{nb_stimuli}.mp4")
        print(f"Fichier créé : {Path(f'tmp_stimulus_{nb_stimuli}.mp4').stat().st_size} bytes")

        print(f"Registered {len(hooks)} hooks")
        with torch.no_grad():
            preds, segments = model.predict(events)

        os.remove(f"tmp_stimulus_{nb_stimuli}.mp4")
        # ── Remove hooks and save ─────────────────────────────────────────────────────
        for h in hooks:
            h.remove()

        print(f"preds shape: {preds.shape}")
        print(f"segments: {segments}")
        all_features_dict = {}
        for layer_name, tensors in features.items():
            stacked = torch.cat(tensors, dim=0)
            safe_name = layer_name.replace('.', '_')
            all_features_dict[safe_name] = stacked.numpy()
            print(f"  {layer_name:40s}  shape={{tuple(stacked.shape)}}")
        file_hdf5 = 'espace_latent.h5'
        with h5py.File(file_hdf5, 'a') as hf:
            group = hf.require_group(f"{subject}/{session}/{run}/stimulus_{nb_stimuli}")
            for safe_name, array in all_features_dict.items():
                group.create_dataset(safe_name, data=array)







    """
    #Tracer preds et segments
    fig, ax = plt.subplots()
    for pred in preds:
        ax.plot(pred, "-")
    plt.show()
    """










