### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from pathlib import Path
from typing import Union, Callable

### External Imports ###
import torch as tc
from torch.nn import Module

### Internal Imports ###
from datasets.wsi import PatchLevelEmbedding, WholeSlideEmbedding

########################


class SlideEncoder(Module):
    """
    Encoder dedicated to convert the patch-level embeddings to a single slide-level embedding.
    """
    def __init__(self,
        encoder : Module
    ):
        """
        Parameters
        ----------
        encoder : Module
            Deep encoder responsible for converting the patch-level embeddings into slide-levle embedding.
        """
        super().__init__()
        self.encoder = encoder
    
    def forward(self,
        embeddings : PatchLevelEmbedding,
        coordinates : tc.Tensor = None,
        metadata : Union[dict, None] = None
    ) -> WholeSlideEmbedding:
        """
        Parameters
        ----------
        embeddings : PatchLevelEmbedding
            Patch-level embeddings to be converted into slide-level embedding
        metadata : dict / None
            Optional metadata associated with the whole slide used by the encoder (e.g. tissue type)
            
        Returns 
        ----------
        embedding : WholeSlideEmbedding
            Embedding representing the WSI at the slide-level
        """
        if metadata is None:
            embedding = self.encoder(embeddings, coordinates)
        else:
            embedding = self.encoder(embeddings, coordinates, metadata)
        return WholeSlideEmbedding(embedding)