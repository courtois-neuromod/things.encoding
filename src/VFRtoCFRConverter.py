import subprocess


class VFRtoCFRConverter:
  """Convertit des vidéos VFR (Variable Frame Rate) en CFR (Constant Frame Rate)."""

  def __init__(self, fps=60, crf=20, video_codec="libx264", overwrite=True):
    """
    Initialise le convertisseur avec les paramètres par défaut.

    Args:
      fps: Nombre d'images par seconde cible (défaut: 60)
      crf: Facteur de qualité CRF, de 18 à 23 (défaut: 20)
      video_codec: Encodeur vidéo à utiliser (défaut: libx264)
      overwrite: Si True, écrase le fichier de sortie s'il existe (défaut: True)
    """
    self.fps = fps
    self.crf = crf
    self.video_codec = video_codec
    self.overwrite = overwrite

  def _build_command(self, input_file, output_file):
    """Construit la commande ffmpeg pour la conversion."""
    command = [
        "ffmpeg",
        "-y" if self.overwrite else "-n",
        "-i", input_file,
        "-c:v", self.video_codec,
        "-crf", str(self.crf),
        "-r", str(self.fps),
        "-c:a", "copy",
        output_file
    ]
    return command

  def convert(self, input_file, output_file):
    """
    Convertit une vidéo VFR en CFR.

    Args:
      input_file: Chemin du fichier vidéo source
      output_file: Chemin du fichier vidéo de sortie

    Returns:
      bool: True si la conversion est réussie, False sinon
    """
    command = self._build_command(input_file, output_file)
    print(f"Début de la conversion de '{input_file}' à {self.fps} images/seconde...")

    try:
      subprocess.run(command, check=True)
      print(f"Succès ! La vidéo CFR a été sauvegardée sous : {output_file}")
      return True
    except subprocess.CalledProcessError as err:
      print(f"Une erreur s'est produite lors de la conversion : {err}")
      return False
    except FileNotFoundError:
      print("Erreur : FFmpeg n'a pas été trouvé. Vérifie qu'il est bien installé.")
      return False

  def set_fps(self, fps):
    """Configure le nombre d'images par seconde."""
    self.fps = fps

  def set_quality(self, crf):
    """Configure le facteur de qualité CRF (18-23)."""
    if not 18 <= crf <= 23:
      raise ValueError("CRF doit être entre 18 et 23")
    self.crf = crf


if __name__ == "__main__":
  converter = VFRtoCFRConverter(fps=64, crf=20)
  source = "../data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1.mp4"
  output = "../things-cfr/sub-01_ses-001_task-thingsmemory_run-1_cfr.mp4"
  converter.convert(source, output)