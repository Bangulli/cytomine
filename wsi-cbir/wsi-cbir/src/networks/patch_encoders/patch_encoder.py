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
from datasets.wsi import PatchLevelEmbedding

########################


class PatchEncoder(Module):
    """
    Encoder dedicated to calculate the patch-level embeddings.
    """
    def __init__(self,
        encoder : Module,
        encoder_transforms : Callable = None
    ):
        """
        Parameters
        ----------
        encoder : Module
            Deep encoder responsible for converting the input patches into embeddings
        encoder_transforms : Callable
            Transforms expected by the deep encoder (to be applied to the loaded tensors)
        """
        super().__init__()
        self.encoder = encoder
        self.encoder_transforms = encoder_transforms
    
    def forward(self,
        images : tc.Tensor,
        metadata : Union[dict, None] = None
    ) -> PatchLevelEmbedding:
        """
        Parameters
        ----------
        images : Tensor
            Patches to convert into embeddings
        metadata : dict / None
            Optional metadata used by the encoder associated with the patches (e.g. pixel size)
            
        Returns
        ---------
        embeddings : PatchLevelEmbedding
            Embeddings representing the input patches
        """
        if self.encoder_transforms is not None: images = self.encoder_transforms(images) 
        if metadata is None:
            embeddings = self.encoder(images)
        else:
            embeddings = self.encoder(images, metadata)
        return PatchLevelEmbedding(embeddings)