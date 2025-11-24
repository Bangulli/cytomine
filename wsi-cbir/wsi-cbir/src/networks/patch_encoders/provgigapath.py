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

### Internal Imports ###

########################

def get_transform():
    return transforms.Compose(
        [
            transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(518),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )


# transform = lambda x: x

class ProvGigaPath_Patch(Module):
    def __init__(self, model_checkpoint_path : Path, jit : bool = False):
        super().__init__()
        if not jit:
            self.encoder = tc.load(model_checkpoint_path, weights_only=False, map_location='cpu')
        else:
            self.encoder = tc.jit.load(model_checkpoint_path, map_location='cpu')
    
    def get_transforms(self):
        return get_transform()
    
    def forward(self, images : tc.Tensor, metadata : dict = None) -> tc.Tensor:
        return self.encoder(images)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
def test_1():
    model = ProvGigaPath_Patch(model_checkpoint_path=Path("../../../models/provgigapath_patch_raw.pth"))
    model.eval()
    batch = tc.randn(8, 3, 518, 518)
    batch = transform(batch)
    output = model(batch)
    print(f"Output shape: {output.shape}")
    
def test_2():
    model = ProvGigaPath_Patch(model_checkpoint_path=Path("../../../models/provgigapath_patch_jit.pth"), jit=True)
    model.eval()
    batch = tc.randn(8, 3, 518, 518)
    batch = transform(batch)
    output = model(batch)
    print(f"Output shape: {output.shape}")
    
    
    
def parse_model_raw():
    encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    raw_model_path = "../../../models/provgigapath_patch_raw.pth"
    tc.save(encoder, raw_model_path)
    print(f"Raw model saved to {raw_model_path}")
    
def parse_model_jit():
    encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
    print(encoder)
    example_images = tc.randn((1, 3, 518, 518), dtype=tc.float32)
    encoder = tc.jit.trace(encoder, example_inputs=[example_images])
    raw_model_path = "../../../models/provgigapath_patch_jit.pth"
    tc.jit.save(encoder, raw_model_path)
    print(f"Raw model saved to {raw_model_path}")
    
    
    
if __name__ == "__main__":
    # test_1()
    # test_2()
    # parse_model_raw()
    # parse_model_jit()
    pass