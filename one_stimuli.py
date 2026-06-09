import os
from pathlib import Path
import pandas as pd
import cv2
from setup import Setup
import ffmpeg

setup = Setup(
        tsv_path=Path("data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1_events.tsv"),
        video_path="data/sub-01/ses-001/sub-01_ses-001_task-thingsmemory_run-1.mp4",
        plateforme="macos"
    )
setup.charger_env()
setup.charger_modele()

tsv_path = setup.tsv_path
video_path = setup.video_path
model = setup.model

df = pd.read_csv(tsv_path, sep="\t")
image_path = df["image_path"]
i = 0

sample = ffmpeg.input(video_path, ss=i, t=2.98).output(f"tmp_stimulus_{i}.mp4", codec="copy").run()
os.remove(f"tmp_stimulus_{i}.mp4")