### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union

### External Imports ###
import numpy as np
import torch as tc
from torch.nn import Module
from torchvision import transforms
import timm
try: from src.networks.CHIEF.models import ctran
except: pass ## dependency issue TODO
### Internal Imports ###

########################

def get_transform():
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    transform = transforms.Compose(
        [
            transforms.Resize(224),
            #transforms.ToTensor(),
            transforms.Normalize(mean = mean, std = std)
        ]
    )
    return transform

# transform = lambda x: x

class CHIEF_Patch(Module):
    def __init__(self, model_checkpoint_path : Path):
        super().__init__()
        self.model = ctran.ctranspath()
        self.model.head = tc.nn.Identity()
        td = tc.load(model_checkpoint_path, map_location='cpu')
        self.model.load_state_dict(td['model'], strict=False)
        self.model.eval()
    
    def get_transforms(self):
        return get_transform()
    
    def forward(self, images : tc.Tensor, metadata : dict = None) -> tc.Tensor:
        assert tc.isfinite(images).all()
        return self.model(images)
    
    
if __name__ == "__main__":
    # test_1()
    # test_2()
    # parse_model_raw()
    # parse_model_jit()
    pass