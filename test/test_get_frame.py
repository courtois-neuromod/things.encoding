import os
import time
import random
from moviepy import VideoFileClip
from PIL import Image


def tester_get_frame(chemin_video, dossier_sortie="../output/test_frames"):
    print(f"--- Début des tests pour : {chemin_video} ---")

    # 1. Chargement de la vidéo
    try:
        video = VideoFileClip(chemin_video)
        print(f"[OK] Vidéo chargée. Durée théorique : {video.duration}s | FPS : {video.fps}")
    except Exception as e:
        print(f"[ERREUR] Impossible de charger la vidéo : {e}")
        return

    os.makedirs(dossier_sortie, exist_ok=True)

    # 4. Validation visuelle (20 captures aléatoires)
    print("\n--- Test : Export de 20 captures aléatoires pour validation visuelle ---")
    nb_captures = 20

    # On tire 20 temps au hasard, en évitant la toute dernière milliseconde
    # On les trie pour optimiser la lecture séquentielle avec ffmpeg
    temps_export = sorted([random.uniform(0, max(0, video.duration - 0.1)) for _ in range(nb_captures)])

    for i, t in enumerate(temps_export):
        try:
            frame = video.get_frame(t)
            img = Image.fromarray(frame)
            nom_fichier = f"capture_rand_{i + 1:02d}_a_{t:.2f}s.jpg"
            chemin_image = os.path.join(dossier_sortie, nom_fichier)
            img.save(chemin_image)
            print(f"  [SUCCÈS] Image {i + 1:02d}/20 exportée : {nom_fichier}")
        except Exception as e:
            print(f"  [ÉCHEC] Impossible d'exporter l'image {i + 1:02d}/20 à {t:.2f}s : {e}")

    video.close()
    print("\n--- Tests terminés ---")


# --- Utilisation ---
if __name__ == "__main__":
    # Chemin de la vidéo à tester
    FICHIER_TEST = "../data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1.mp4"

    if os.path.exists(FICHIER_TEST):
        tester_get_frame(FICHIER_TEST)
    else:
        print(f"Veuillez définir un chemin de vidéo valide (actuellement : {FICHIER_TEST})")