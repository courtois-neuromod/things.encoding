"""
Extraction des représentations latentes de TRIBE v2 sur toutes les vidéos de friends.
Parcourt le dossier de stimuli d'une saison, passe chaque vidéo à TribeModel avec
forward hooks, et sauvegarde les activations en HDF5.
"""
import warnings
import argparse
import logging
import h5py
import torch
from pathlib import Path
import gc
import time

from TransformerHooks import TransformerHooks
from Config import Config
from HDF5Writer import HDF5Writer

def sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extraction des latents TRIBE v2')
    parser.add_argument('--season', type=int, required=True, help='Identifiant de la saison à traiter')
    parser.add_argument('--episode', type=int, required=True, help="Identifiant de l'episode à traiter")
    args = parser.parse_args()
    season = args.season
    episode = args.episode

    #warnings.filterwarnings("ignore")
    #logging.disable(logging.CRITICAL)

    plateforme = ['Rorqual', 'Mac']
    plateforme = plateforme[0]

    if plateforme == "Rorqual":
        ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
        ROOT_STIMULI = Path(f"/home/aclaud/links/scratch/friends.stimuli/s{season}")
    else:
        ROOT_ENCODING = Path(__file__).parent.parent
        ROOT_STIMULI = ROOT_ENCODING / "friends.stimuli" / f"s{season}"

    DATA_DIR = ROOT_STIMULI
    HDF5_DIR = ROOT_ENCODING / "output" / "hdf5" / "friends"

    config = Config(
        plateforme=plateforme,
    )
    config.charger_env()
    print(f"Env chargé", flush=True)

    t0 = sync_time()
    model = config.charger_modele()
    t_charge = sync_time() - t0
    print(f"Modèle chargé en {t_charge:.2f} secondes", flush=True)
    fmri_enc = model.__pydantic_private__['_model']

    t1 = sync_time()
    writer = HDF5Writer(HDF5_DIR)
    video_files = sorted(DATA_DIR.glob(f"friends_s{season:02d}e{episode:02d}[abc].mkv"))
    t_recherche_video = sync_time() - t1
    print(f"Saison {season} --> {len(video_files)} vidéos trouvées en {t_recherche_video:.2f} secondes", flush=True)

    t_script_start = sync_time()
    for video_path in video_files:
        episode_str = video_path.stem.split("_")[-1]  # ex. "s01e01a"

        output_path = HDF5_DIR / f"{season}.h5"
        run_path = f"{episode_str}/clip"
        if output_path.exists():
            with h5py.File(output_path, "r") as hf:
                if run_path in hf and 'preds' in hf[run_path]:
                    print(f"{episode_str} déjà traité")
                    continue
        try:
            t_vid_start = sync_time()
            events = model.get_events_dataframe(video_path=str(video_path))

            t_hook_start = sync_time()
            hooks = TransformerHooks(fmri_enc)
            hooks.attacher()
            t_hook = sync_time() - t_hook_start
            print(f"Hook chargé en {t_hook:.2f} secondes", flush=True)

            t_infer_start = sync_time()
            with torch.no_grad():
                preds, segments = model.predict(events)
            t_infer = sync_time() - t_infer_start
            print(f"Inférence réalisé en {t_infer:.2f} secondes", flush=True)

            hooks.retirer()
            features = hooks.get_features()

            t_enregistrement_start = sync_time()
            writer.sauvegarder(features, preds, str(season), episode_str, "clip")
            t_enregistrement = sync_time() - t_enregistrement_start
            print(f"Video enregistré en {t_enregistrement:.2f} secondes", flush=True)

            t_vid_total = sync_time() - t_vid_start
            print(f"{episode_str} traité en {t_vid_total:.2f} secondes - preds shape: {preds.shape}", flush=True)

            # Nettoyage de la mémoire pour la vidéo suivante
            del features, preds, segments, events, hooks

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"{episode_str} non traité - Erreur: {e}", flush=True)
    t_script_total = sync_time() - t_script_start
    print(f"Temps total pour la boucle: {t_script_total:.2f} secondes", flush=True)
