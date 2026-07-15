"""
Extraction des représentations latentes de TRIBE v2 sur toutes les vidéos de friends.
Parcourt le dossier de stimuli d'une saison, passe chaque vidéo à TribeModel avec
forward hooks, et sauvegarde les activations en HDF5.
"""
import warnings
import argparse
import logging
from pathlib import Path


from Config import Config
from HDF5Writer import HDF5Writer

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extraction des latents TRIBE v2')
    parser.add_argument('--season', type=int, required=True, help='Identifiant de la saison à traiter')
    args = parser.parse_args()
    season = args.season

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

    model = config.charger_modele()
    print(f"Modèle chargé", flush=True)
    fmri_enc = model.__pydantic_private__['_model']

    writer = HDF5Writer(HDF5_DIR)
    video_files = sorted(DATA_DIR.glob(f"friends_s{season:02d}e*.mkv"))
    print(f"Saison {season} --> {len(video_files)} vidéos trouvées")
    """
for video_path in video_files:
    episode = video_path.stem.split("_")[-1]  # ex. "s01e01a"

    output_path = HDF5_DIR / f"{season}.h5"
    run_path = f"{episode}/clip"
    if output_path.exists():
        with h5py.File(output_path, "r") as hf:
            if run_path in hf and 'preds' in hf[run_path]:
                print(f"{episode} déjà traité")
                continue

    try:
        events = model.get_events_dataframe(video_path=str(video_path))

        hooks = TransformerHooks(fmri_enc)
        hooks.attacher()

        with torch.no_grad():
            preds, segments = model.predict(events)

        hooks.retirer()
        features = hooks.get_features()

        writer.sauvegarder(features, preds, str(season), episode, "clip")

        print(f"{episode} traité - preds shape: {preds.shape}", flush=True)

        # Nettoyage de la mémoire pour la vidéo suivante
        del features, preds, segments, events, hooks

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"{episode} non traité - Erreur: {e}", flush=True)
"""