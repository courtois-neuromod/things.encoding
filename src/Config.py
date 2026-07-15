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
        token = os.environ.get("HF_TOKEN")
        if token and not os.environ.get("HF_HUB_OFFLINE"):
            login(token=token)
        else:
            print("Mode offline : login HuggingFace ignoré, utilisation du cache.")

    def charger_modele(self):
        if self.plateforme == ('Mac'):
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
