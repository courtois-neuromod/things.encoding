import subprocess
from pathlib import Path
import pandas as pd


def _build_command(onset, video_name, output_image_name):
    """Construit la commande ffmpeg pour la conversion."""
    command = [
        "ffmpeg",
        "-ss", str(onset),
        "-i", str(video_name),
        "-vframes", "1",
        str(output_image_name)
    ]
    return command

plateforme = ['Rorqual', 'Mac']
plateforme = plateforme[0]

if plateforme == "Rorqual":
    ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
    PATH_VIDEO = ROOT_ENCODING / "data" / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1.mp4"
    PATH_TSV = ROOT_ENCODING / "data" / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1_events.tsv"
else:
    ROOT_ENCODING = Path(__file__).parent.parent
    PATH_VIDEO = ROOT_ENCODING / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1.mp4"
    PATH_TSV = ROOT_ENCODING / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1_events.tsv"


DATA = PATH_VIDEO
OUTPUT_DIR = ROOT_ENCODING / "output" / "OneImageVideo"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PATH_TSV, sep='\t')

image = df.iloc[0, 1]
onset = df.iloc[0, 17]
offset = df.iloc[0, 18]
image_name = "image1.png"
OUTPUT_IMAGE = OUTPUT_DIR / image_name
command = _build_command(onset, PATH_VIDEO, OUTPUT_IMAGE)
subprocess.run(command, check=True)

