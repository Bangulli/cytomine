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

try: from CHIEF.models.CHIEF import CHIEF
except: pass ## dependency issue TODO

########################


class CHIEF_Slide(Module):
    def __init__(self, model_checkpoint_path : Path):
        super().__init__()
        self.encoder = CHIEF(size_arg="small", dropout=True, n_classes=2)
        self.encoder.load_state_dict(tc.load(model_checkpoint_path, map_location='cpu'), strict=False)
        self.encoder.eval()
    
    def forward(self, embeddings : tc.Tensor, coordinates : tc.Tensor, metadata : dict = None) -> tc.Tensor:
        assert tc.isfinite(embeddings).all()
        emb = self.encoder(embeddings, coordinates)[0]
        assert tc.isfinite(emb).all()
        return emb
    
    
if __name__ == "__main__":

    pass