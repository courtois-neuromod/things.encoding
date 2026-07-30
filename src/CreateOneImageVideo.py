from pathlib import Path
import pandas as pd

plateforme = ['Rorqual', 'Mac']
plateforme = plateforme[0]

if plateforme == "Rorqual":
    ROOT_ENCODING = Path("/home/aclaud/links/scratch/things.encoding")
    PATH_IMAGE = ROOT_ENCODING / "data" / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1.mp4"
    PATH_TSV = ROOT_ENCODING / "data" / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1_events.tsv"
else:
    ROOT_ENCODING = Path(__file__).parent.parent
    PATH_IMAGE = ROOT_ENCODING / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1.mp4"
    PATH_TSV = ROOT_ENCODING / "data" / "sub-01" / "ses-001" / "sub-01_ses-001_task-thingsmemory_run-1_events.tsv"


DATA = PATH_IMAGE
OUTPUT_DIR = ROOT_ENCODING / "output" / "OneImageVideo"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PATH_TSV, sep='\t')
