### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union
import json
### External Imports ###
import numpy as np
import torch as tc
from torch.nn import Module
from torchvision import transforms
import timm
from transformers import AutoModel

### Internal Imports ###
from src.networks.TITAN.conch_v1_5 import build_conch
from src.networks.TITAN.configuration_titan import ConchConfig
########################


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(448, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True, max_size=None),
            transforms.CenterCrop(448),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )

class TITAN_Patch(Module):
    def __init__(self, model_checkpoint_path : Path):
        super().__init__()
        self.encoder = build_conch(ConchConfig(), model_checkpoint_path)
        
    def get_transforms(self):
        return get_transform()
    
    def forward(self, images : tc.Tensor, metadata : dict = None) -> tc.Tensor:
        with tc.inference_mode(): return self.encoder(images)
