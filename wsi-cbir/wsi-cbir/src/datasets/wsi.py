### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union, Tuple, Callable
from dataclasses import dataclass

### External Imports ###
import torch as tc
from torch.utils.data import Dataset

### Internal Imports ###

########################


@dataclass
class WholeSlideMetadata():
    """
    Dataclass with patch-level and slide-level metadata as dictionaries.
    """
    patch_metadata : dict
    slide_metadata : dict
    
class PatchLevelEmbedding(tc.Tensor):
    """
    Represents the patch-level embedding
    """
    def concatenate(self, other : 'PatchLevelEmbedding') -> 'PatchLevelEmbedding':
        self = tc.cat((self, other), dim=0)
        return self


class WholeSlideEmbedding(tc.Tensor):
    """
    Represents the slide-level embedding.
    """
    def load_embedding(self, path : Union[str, Path]):
        self = tc.load(path, map_location="cpu", weights_only=False)
        return self
        
    def save_embedding(self, path : Union[str, Path]):
        tc.save(self, path)
    
# class WholeSlideEmbeddings(Dataset):
#     """
#     Represents a dataset of slide-level embeddings.
#     """
#     def __init__(self,
#         embedding_paths
#     ):
#         self.embedding_paths = embedding_paths
#         # TODO: funcionality to load from BigPicture repository
    
#     def __len__(self):
#         return len(self.embedding_paths)
    
#     def __getitem__(self, idx):
#         embedding = WholeSlideEmbedding()
#         embedding = embedding.load_embedding(self.embedding_paths[idx])
#         return embedding
    
class WholeSlide(Dataset):
    """
    Parent class (to overload) responsible for managing the whole slide images. It handles masking the WSI content and splits the image into patches during loading.
    """
    def __init__(self,
        wsi_path : Union[str, Path],
        mask_path : Union[str, Path, None],
        metadata_path : Union[str, Path, None],
        resolution_level : int,
        patch_size : tuple,
        patch_stride: tuple,
        return_patch_metadata : bool = False,
        return_slide_metadata : bool = False,
        calculate_mask : bool = False,
        calculate_mask_params : Union[dict, None] = None):
        transforms : Callable = None,
        """
        
        """
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.metadata_path = metadata_path
        self.resolution_level = resolution_level
        self.patch_size = patch_size
        self.patch_overlap = patch_stride
        self.return_patch_metadata = return_patch_metadata
        self.return_slide_metadata = return_slide_metadata
        self.calculate_mask = calculate_mask
        self.calculate_mask_params = calculate_mask_params
        self.transforms = transforms
    
    def __len__(self) -> int:
        raise NotImplementedError()
    
    def __getitem__(self, idx : int) -> Tuple[tc.Tensor, Union[None, WholeSlideMetadata]]:
        raise NotImplementedError()
    
    def get_resolution(self):
        raise NotImplementedError()

    def get_number_of_rows(self):
        raise NotImplementedError()

    def get_number_of_cols(self):
        raise NotImplementedError()
    

