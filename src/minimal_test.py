"""
Script de test minimal : prédictions BOLD + latents du transformer FmriEncoder
sur le RUN CONTINU (pas les micro-clips).
"""
import warnings
import logging
from pathlib import Path
from collections import defaultdict

import torch

from Config import Config

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    logging.disable(logging.CRITICAL)

    ROOT = Path(__file__).parent.parent
    VIDEO_PATH = ROOT / "data/sub-01/ses-001/lite_run_for_test.mp4"

    # --- Chargement du modèle ---
    setup = Config(plateforme="macos")
    setup.charger_env()
    model = setup.charger_modele()

    # --- Lecture du TR  ---
    print("data frequency :", model.data.neuro.frequency)
    print("tribe TR       :", model.data.TR)

    fmri_enc = model.__pydantic_private__["_model"]
    print(fmri_enc.config.linear_baseline)

    print("overlap_trs_train : ", model.data.overlap_trs_train)
    print("data.duration_trs : ", model.data.duration_trs, "/ TR : ", model.data.TR)

    # --- Inspection de la structure des couches (à vérifier une fois) ---
    """
    print("\nStructure encoder.layers :")
    for idx, block in enumerate(fmri_enc.encoder.layers):
        contenu = (
            [type(m).__name__ for m in block]
            if hasattr(block, "__iter__")
            else type(block).__name__
        )
        print(f"  [{idx}] {contenu}")
    """
    # --- Pose des hooks sur les blocs d'attention ---
    features = defaultdict(list)
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            tensor = out[0] if isinstance(out, tuple) else out
            features[name].append(tensor.detach().cpu())
            print("latent shape (B, T, H):", tensor.shape)
        return hook

    for idx, block in enumerate(fmri_enc.encoder.layers):
        if idx % 2 == 0:  # blocs d'attention (à confirmer via l'inspection ci-dessus)
            name = f"encoder.layer{idx // 2}.attn"
            hooks.append(block[1].register_forward_hook(make_hook(name)))
    print(f"\nHooks enregistrés : {len(hooks)}")

    # --- video ---
    events = model.get_events_dataframe(video_path=str(VIDEO_PATH))
    print("\nEvents :")
    print(events)

    with torch.no_grad():
        preds, segments = model.predict(events)

    print(f"\npreds shape    : {preds.shape}")
    print(f"n segments     : {len(segments)}")

    # --- Récupération des latents capturés ---
    print("\nLatents transformer :")
    for name, tensors in features.items():
        stacked = torch.cat(tensors, dim=0)
        print(f"  {name:30s} shape={tuple(stacked.shape)}")

    # --- Nettoyage ---
    for h in hooks:
        h.remove()