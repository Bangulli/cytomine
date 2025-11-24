### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

### External Imports ###
import torch as tc
from torch.nn import Module

### Internal Imports ###

from gigapath import slide_encoder as se

########################


class ProvGigaPath_Slide(Module):
    def __init__(self, model_checkpoint_path : Path):
        super().__init__()
        self.encoder = se.gigapath_slide_enc12l768d()
        self.encoder.load_state_dict(tc.load(model_checkpoint_path, map_location='cpu')["model"], strict=False)
    
    def forward(self, embeddings : tc.Tensor, coordinates : tc.Tensor, metadata : dict = None) -> tc.Tensor:
        return self.encoder(embeddings, coordinates)[0]
    
    
def test_1():
    device = "cuda:0"
    model = ProvGigaPath_Slide(model_checkpoint_path=Path("../../../models/provgigapath_slide.pth"))
    model.eval()
    model = model.to(device)
    embeddings = tc.randn(1, 8, 1536).to(device)
    coordinates = tc.randn(1, 8, 2) .to(device)
    output = model(embeddings, coordinates)
    print(f"Output shape: {output.shape}")




    
    
if __name__ == "__main__":
    test_1()
    pass