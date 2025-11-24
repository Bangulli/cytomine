### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path

### External Imports ###
import torch as tc
from transformers import AutoModel
from torch.nn import Module
from safetensors.torch import load_file, safe_open
### Internal Imports ###
from src.networks.PRISM import modeling_prism
from src.networks.PRISM import configuring_prism
########################


class PRISM_Slide(Module):
    def __init__(self, model_checkpoint_path : Path):
        super().__init__()
        self.prism = modeling_prism.Prism(configuring_prism.PrismConfig())
        
        self.prism.load_state_dict(tc.load(model_checkpoint_path, map_location='cpu'), strict=True)
    
    def forward(self, embeddings : tc.Tensor, coordinates : tc.Tensor, metadata : dict = None) -> tc.Tensor:
        assert tc.isfinite(embeddings).all()
        emb = self.prism.slide_representations(embeddings)['image_embedding']
        assert tc.isfinite(emb).all()
        return emb
    
    
if __name__ == "__main__":

    pass