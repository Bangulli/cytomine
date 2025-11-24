### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union
import json
import subprocess
### External Imports ###
import numpy as np
import torch as tc
from torch.nn import Module
from torchvision import transforms
import timm
from timm.layers import SwiGLUPacked
from transformers import AutoModel

### Internal Imports ###

########################


def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True, max_size=None),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )



class PRISM_Patch(Module):
    def __init__(self, model_checkpoint_path : Path):
        super().__init__()
        self.encoder = timm.create_model("local-dir:./src/networks/Virchow", mlp_layer=SwiGLUPacked, act_layer=tc.nn.SiLU, checkpoint_path=model_checkpoint_path)
        
    def get_transforms(self):
        return get_transform()
    
    def forward(self, images : tc.Tensor, metadata : dict = None) -> tc.Tensor:
        with tc.inference_mode(): 
            output = self.encoder(images) 
            class_token = output[:, 0]    # size: 1 x 1280
            patch_tokens = output[:, 1:]  # size: 1 x 256 x 1280

            # concatenate class token and average pool of patch tokens
            embedding = tc.cat([class_token, patch_tokens.mean(1)], dim=-1)  # size: 1 x 2560
            return embedding
