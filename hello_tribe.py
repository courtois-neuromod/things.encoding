"""
Trouve le nn.Module PyTorch dans TribeModel via _init_module et le pl_module.
Lance avec : uv run python hello_tribe.py 2>/dev/null
"""
import torch
import torch.nn as nn
import os
from dotenv import load_dotenv
from tribev2 import TribeModel
from huggingface_hub import login

# charge le .env automatiquement
load_dotenv()  

login(token=os.environ["HF_TOKEN"])

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
