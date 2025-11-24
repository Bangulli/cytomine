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
from safetensors.torch import load_file
### Internal Imports ###
from src.networks.TITAN import modeling_titan
from src.networks.TITAN.configuration_titan import TitanConfig
########################


# transform = lambda x: x

class TITAN_Slide(Module):
    def __init__(self, model_checkpoint_path : Path):
        super().__init__()
        self.titan = modeling_titan.Titan(TitanConfig())
        state_dict = load_file(model_checkpoint_path)
        self.titan.load_state_dict(state_dict, strict=True)
    
    def forward(self, patches : tc.Tensor, coords : tc.Tensor) -> tc.Tensor:
        with tc.inference_mode(): return self.titan.encode_slide_from_patch_features(patches.type(tc.float32), coords.type(tc.long), tc.tensor(518))
