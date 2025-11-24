### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union
import time

### External Imports ###
import torch as tc
from torch.nn import Module
from torch.utils.data import DataLoader
from tqdm import tqdm
### Internal Imports ###
from datasets.wsi import WholeSlide, WholeSlideMetadata, PatchLevelEmbedding, WholeSlideEmbedding
from networks.patch_encoders.patch_encoder import PatchEncoder
from networks.slide_encoders.slide_encoder import SlideEncoder

########################


def calculate_patch_level_embeddings(
    wsi : WholeSlide,
    patch_encoder : PatchEncoder,
    batch_size : int,
    num_workers : int = 0,
    device : Union[str, tc.device] = 'cpu',
    echo : bool = False,
    ) -> PatchLevelEmbedding:
    """
    Calculates the patch-level embeddings using image-only or combination of image and image metadata (e.g. pixel size).
    
    Parameters
    ----------
    wsi : WholeSlide
        An instance of WholeSlide dataset to load the patches.
    patch_encoder : PatchEncoder
        Deep encoder used to calculate patch-level embeddings
    batch_size : int
        The number of patches to process simultaneously
    """
    dataloader = DataLoader(wsi, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    for idx, batch in enumerate(dataloader):
        with tc.no_grad():
            images = batch['images']
            coordinates = batch['coordinates']
            coordinates = coordinates.to(device, non_blocking=True)
            images = images.to(device)
            ### Calculate Embeddings ###
            if wsi.return_patch_metadata:
                metadata : WholeSlideMetadata = batch['metadata']
                embeddings = patch_encoder(images, metadata.patch_metadata)
            else:
                embeddings = patch_encoder(images)
            ### Accumulate Embeddings ###
            if idx == 0:
                patch_embeddings : PatchLevelEmbedding = embeddings
                patch_coordinates : tc.Tensor = coordinates
                if wsi.return_slide_metadata:
                    metadata : WholeSlideMetadata = batch['metadata']
                    slide_metadata = metadata.slide_metadata
            else:
                patch_embeddings = patch_embeddings.concatenate(embeddings)
                patch_coordinates = tc.cat((patch_coordinates, coordinates), dim=0)
            
    patch_coordinates = patch_coordinates.to(device).unsqueeze(0).to(tc.float32)
    if wsi.return_slide_metadata:
        return patch_embeddings, patch_coordinates, slide_metadata
    else:
        return patch_embeddings, patch_coordinates
    

    
def calculate_slide_level_embedding(
    wsi : WholeSlide,
    patch_encoder : PatchEncoder,
    slide_encoder : SlideEncoder,
    batch_size : int,
    num_workers : int = 0,
    device: Union[str, tc.device] = 'cpu',
    echo : bool = False,
    ) -> WholeSlideEmbedding:
    """
    Calculates the slide-level embedding using image-only or combination of image and image metadata (e.g. specie, tissue type).
    
    Parameters
    ----------
    wsi : WholeSlide
        An instance of WholeSlide dataset to load the patches.
    patch_encoder : PatchEncoder
        Deep encoder used to calculate patch-level embeddings
    slide_encoder : SlideEncoder
        Deep encoder to calculate the slide-level embedding usign the patch-level embeddings
    batch_size : int
        The number of patches to process simultaneously
    """
    with tc.no_grad():
        if wsi.return_slide_metadata:
            patch_embeddings, patch_coordinates, slide_metadata = calculate_patch_level_embeddings(wsi, patch_encoder, batch_size, num_workers=num_workers, device=device, echo=echo)
            slide_embedding = slide_encoder(patch_embeddings.to(device).unsqueeze(0), patch_coordinates.to(device), slide_metadata.to(device))[0, :]
        else:
            patch_embeddings, patch_coordinates = calculate_patch_level_embeddings(wsi, patch_encoder, batch_size, num_workers=num_workers, device=device, echo=echo)
            slide_embedding = slide_encoder(patch_embeddings.to(device).unsqueeze(0), patch_coordinates.to(device))[0, :]
    return slide_embedding