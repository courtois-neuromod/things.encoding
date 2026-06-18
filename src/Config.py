import os
from dotenv import load_dotenv
from tribev2 import TribeModel
from huggingface_hub import login

class Config():

    def __init__(self, plateforme):
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
		device="cuda",
                config_update={"data.video_feature.image.device": "cuda"},
            )
        return self.model
