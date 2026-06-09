import os
from dotenv import load_dotenv
from tribev2 import TribeModel
from huggingface_hub import login

class Setup():

    def __init__(self, tsv_path, video_path, plateforme):
        self.tsv_path = tsv_path
        self.video_path = video_path
        self.plateforme = plateforme

    def definir_plateforme(self, plateforme):
        self.plateforme = plateforme

    def charger_env(self):
        load_dotenv()
        login(token=os.environ["HF_TOKEN"])

    def charger_modele(self):
        if self.plateforme == 'macos':
            self.model = TribeModel.from_pretrained(
                "facebook/tribev2",
                cache_folder="./cache",
                device="cpu",
                config_update={"data.video_feature.image.device": "cpu"},
            )

        else:
            self.model = TribeModel.from_pretrained(
                "facebook/tribev2",
                cache_folder="./cache",
            )

    def get_tsv_path(self):
        return self.tsv_path

    def get_video_path(self):
        return self.video_path
