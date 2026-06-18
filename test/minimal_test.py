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
    VIDEO_PATH = "sub-01_ses-001_task-thingsmemory_run-1_cfr.mp4"

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

    print("\nAnalyse temporelle des segments :")
    for i, seg in enumerate(segments):
        # On extrait les temps réels du segment (à vérifier selon la structure exacte renvoyée par TRIBE)
        start_time = getattr(seg, 'start', i * 100) # Fallback théorique si l'attribut diffère
        print(f"  Segment {i:02d} : Chronologie vidéo = {start_time} s")

    # --- CORRECTION : Récupération intelligente des latents ---
    print("\nLatents transformer (Cartographie temporelle) :")
    for name, tensor_list in features.items():
        print(f"\nCouche : {name}")

        # tensor_list contient autant de tenseurs que de segments traités
        # Au lieu de tout écraser avec torch.cat, on les associe à leur segment
        for i, tensor in enumerate(tensor_list):
            seg_start = getattr(segments[i], 'start', '???')
            # Le tenseur a la forme (B, T, H) où T est à 2 Hz (0.5s)
            print(f"  -> Appartient au Segment {i} (Début: {seg_start} s) | Shape: {tuple(tensor.shape)}")

        # Si tu as besoin de tout empiler à la fin pour l'Étape 3 :
        stacked = torch.cat(tensor_list, dim=0)
        print(f"  Shape totale empilée = {tuple(stacked.shape)}")

    # --- Nettoyage ---
    for h in hooks:
        h.remove()
