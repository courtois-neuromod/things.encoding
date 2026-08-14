from pathlib import Path

import pandas as pd

from VideoSegmenteur import VideoSegmenteur

plateforme = ["Rorqual", "Mac"]
plateforme = plateforme[0]  # [0] pour Rorqual, [1] pour Mac

if plateforme == "Rorqual":
    ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
    # Sur Rorqual, les TSV sont dans data/data/sub-XX/...
    DATA_DIR = ROOT_ENCODING / "data" / "data"
    # Dossier des vidéos CFR sur Rorqual
    CFR_DIR = ROOT_ENCODING / "data" / "things_mp4_cfr"
else:
    ROOT_ENCODING = Path(__file__).parent.parent
    # Sur Mac, les TSV sont dans data/things_mp4_vfr/sub-XX/...
    DATA_DIR = ROOT_ENCODING / "data" / "things_mp4_vfr"
    # Dossier des vidéos CFR sur Mac
    CFR_DIR = ROOT_ENCODING / "data" / "things_mp4_cfr"

OUTPUT_DIR = ROOT_ENCODING / "output" / "analysis" / "OneImageVideo"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Récupère tous les .tsv récursivement dans les sous-dossiers
tsv_files = sorted(DATA_DIR.rglob("*_events.tsv"))

print(f"{len(tsv_files)} fichiers TSV trouvés au total sur {plateforme}.")

# -- Étape 1 : construction de la liste des images distinctes --
# Pour chaque image distincte, on garde sa première occurrence (onset, offset, vidéo source)
images_distinctes = {}

for path_tsv in tsv_files:
    base_name = path_tsv.name.replace("_events.tsv", "")
    path_video = CFR_DIR / f"{base_name}_desc-CFR.mp4"

    # Vérification si la vidéo source existe bien
    if not path_video.exists():
        print(f"Vidéo introuvable : {path_video.name}, on passe au suivant.")
        continue

    df = pd.read_csv(path_tsv, sep="\t")

    for _idx, row in df.iterrows():
        valeur_image = str(row.iloc[1])
        if pd.isna(valeur_image) or valeur_image == "nan":
            continue

        image_name = Path(valeur_image).stem

        if image_name in images_distinctes:
            continue

        onset = row.iloc[17]
        offset = row.iloc[18]
        images_distinctes[image_name] = (onset, offset, path_video)

print(f"{len(images_distinctes)} images distinctes trouvées.\n")

# -- Étape 2 : création d'une vidéo de 100s par image distincte --
for image_name, (onset, offset, path_video) in images_distinctes.items():
    nom_fichier_100s = f"{image_name}_100s.mp4"
    output_video_path = OUTPUT_DIR / nom_fichier_100s

    if output_video_path.exists():
        print(f"Vidéo déjà créée, on passe : {nom_fichier_100s}")
        continue

    duration = offset - onset - 0.5
    nom_fichier_3s = f"{image_name}.mp4"
    input_video_path = str(OUTPUT_DIR / nom_fichier_3s)

    segmenteur = VideoSegmenteur(path_video, OUTPUT_DIR)
    segmenteur.create_segment(onset, duration, input_video_path)
    segmenteur.etendre_video(100, input_video_path, str(output_video_path))
    print(f"Vidéo créée : {nom_fichier_100s}")
