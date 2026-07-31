import subprocess
from pathlib import Path
import pandas as pd
from VideoSegmenteur import VideoSegmenteur

plateforme = ['Rorqual', 'Mac']
plateforme = plateforme[0]  # [0] pour Rorqual, [1] pour Mac

if plateforme == "Rorqual":
    ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
    # Sur Rorqual, les TSV sont dans data/data/sub-XX/...
    DATA_DIR = ROOT_ENCODING / "data" / "data"
    # Dossier des vidéos CFR sur Rorqual
    CFR_DIR = ROOT_ENCODING / "data" / "things_mp4_cfr"
else:
    ROOT_ENCODING = Path(__file__).parent.parent
    # Sur Mac, les TSV sont directement dans data/sub-XX/...
    DATA_DIR = ROOT_ENCODING / "data"
    # Dossier des vidéos CFR sur Mac
    CFR_DIR = ROOT_ENCODING / "data" / "things_mp4_cfr"

OUTPUT_DIR = ROOT_ENCODING / "output" / "OneImageVideo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Récupère tous les .tsv récursivement dans les sous-dossiers
tsv_files = sorted(DATA_DIR.rglob("*_events.tsv"))

print(f"{len(tsv_files)} fichiers TSV trouvés au total sur {plateforme}.")

compteur_images = {}

for path_tsv in tsv_files:
    base_name = path_tsv.name.replace("_events.tsv", "")

    # -- Choix de la vidéo --
    if plateforme == "Rorqual":

        path_video = CFR_DIR / f"{base_name}_desc-CFR.mp4"
    else:
        path_video = CFR_DIR / f"{base_name}_desc-CFR.mp4"

    # Vérification si la vidéo source existe bien
    if not path_video.exists():
        print(f"Vidéo introuvable : {path_video.name}, on passe au suivant.")
        continue

    print(f"\n Traitement de : {base_name}")

    df = pd.read_csv(path_tsv, sep='\t')
    segmenteur = VideoSegmenteur(path_video, OUTPUT_DIR)

    for idx, row in df.iterrows():
        onset = row.iloc[17]
        offset = row.iloc[18]
        duration = offset - onset - 0.5

        valeur_image = str(row.iloc[1])
        if pd.isna(valeur_image) or valeur_image == "nan":
            continue

        nom_stem = Path(valeur_image).stem
        if '_' in nom_stem:
            image_name = nom_stem.rsplit('_', 1)[0]
        else:
            image_name = nom_stem

        compteur_images[image_name] = compteur_images.get(image_name, 0) + 1
        occurrence_actuelle = compteur_images[image_name]

        if occurrence_actuelle == 1:
            nom_fichier_3s = f"{image_name}.mp4"
            nom_fichier_100s = f"{image_name}_100s.mp4"
        else:
            nom_fichier_3s = f"{image_name}__{occurrence_actuelle}.mp4"
            nom_fichier_100s = f"{image_name}_{occurrence_actuelle}_100s.mp4"

        input_video_path = str(OUTPUT_DIR / nom_fichier_3s)
        output_video_path = str(OUTPUT_DIR / nom_fichier_100s)

        if Path(output_video_path).exists():
            continue

        segmenteur.create_segment(onset, duration, input_video_path)
        segmenteur.etendre_video(100, input_video_path, output_video_path)