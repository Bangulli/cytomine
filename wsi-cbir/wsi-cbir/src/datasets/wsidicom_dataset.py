### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union, Tuple, Callable
import time
import logging ## avoid weird wsidicom logspam: WARNING:root:Orientation [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0] is not orthogonal with equal lengths with column rotated 90 deg from row
logging.getLogger().setLevel(logging.ERROR)
### External Imports ###
import numpy as np
import torch as tc
from wsidicom import WsiDicom
import skimage
### Internal Imports ###
import wsi
from src.datasets.bg_removal import get_bg_rm_tool
########################

class WsiDicomDataset(wsi.WholeSlide):
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
        calculate_mask_params : str = 'dilated_otsu',
        transforms : Callable = None,
        half_precision : bool = False,
        ):
        """
        
        """
        self.wsi_path = wsi_path
        self.mask_path = mask_path
        self.metadata_path = metadata_path
        self.resolution_level = resolution_level
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.return_patch_metadata = return_patch_metadata
        self.return_slide_metadata = return_slide_metadata
        self.calculate_mask = calculate_mask
        self.calculate_mask_params = calculate_mask_params
        self.transforms = transforms
        self.half_precision = half_precision
        self.gs_threshold = None
        self.image_slide = None

        if self.metadata_path is not None:
            pass # TODO - handle external metadata

        image_slide = WsiDicom.open(self.wsi_path)
        assert self.resolution_level<len(image_slide.levels), f"User requested level {self.resolution_level}, but the loaded DICOM WSI only has {len(image_slide.levels)} levels: {image_slide.levels}"

        mask_slide = get_bg_rm_tool(self.calculate_mask_params)(image_slide) if self.calculate_mask else None
            
        self.mpp = (image_slide.levels[self.resolution_level].mpp.width, image_slide.levels[self.resolution_level].mpp.height)
        
        self.upper_left_corners, self.coordinates = self.calculate_upper_left_corners(image_slide, mask_slide)
        self.resolution = (image_slide.levels[self.resolution_level].size.height, image_slide.levels[self.resolution_level].size.width)

        image_slide.close()
        
    def calculate_upper_left_corners(self, image_slide : WsiDicom, mask_slide : WsiDicom=None):
        """Calculates the coordinates of the upper left corners of every patch in the slide given the patch stride and patch size arguments
        Args:
            image_slide (WsiDicom): _description_
            mask_slide (WsiDicom, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        height = image_slide.levels[self.resolution_level].size.height 
        width = image_slide.levels[self.resolution_level].size.width 
        height_multiplier = 1 if self.resolution_level == 0 else int(height / image_slide.levels[self.resolution_level].size.height)
        width_multiplier = 1 if self.resolution_level == 0 else int(width / image_slide.levels[self.resolution_level].size.width)
        step_height = self.patch_stride[0] * height_multiplier
        step_width = self.patch_stride[1] * width_multiplier
        corners = []
        coordinates = []
        self.number_of_rows = len(list(range(0, height - step_height, step_height)))
        self.number_of_cols = len(list(range(0, width - step_width, step_width)))
        for y in range(0, height - step_height, step_height):
            for x in range(0, width - step_width, step_width):
                coordinate = np.array([int(x), int(y)])
                coordinate_given_level = np.array([int(x / width_multiplier), int(y / height_multiplier)])
                
                if mask_slide is not None:
                    mask_patch = mask_slide.read_region((coordinate[0], coordinate[1]), self.resolution_level, self.patch_size)
                    if np.any(mask_patch):
                        corners.append(coordinate)
                        coordinates.append(coordinate_given_level)
                        
                else:
                    corners.append(coordinate)
                    coordinates.append(coordinate_given_level)
        corners = np.array(corners)
        print(f"=== Obtained {corners.shape[0]} patches for image")
        coordinates = np.array(coordinates)
        return corners, coordinates
    
    def _ensure_image_is_open(self):
        """Ensures that an instance of the image is opened inside the worker to avoid errors from multiple workers accessing the same instance at the same time leading to errors
        """
        if self.image_slide is None:
            self.image_slide = WsiDicom.open(self.wsi_path)
            
    def load_patch_at(self, coordinates):
        self._ensure_image_is_open()
        patch = self.image_slide.read_region(coordinates, self.resolution_level, self.patch_size).convert('RGB')
        patch = np.array(patch)
        return patch

    def get_coordinates(self, idx):
        return self.upper_left_corners[idx], self.coordinates[idx]
    
    def get_resolution(self):
        return self.resolution

    def get_number_of_rows(self):
        return self.number_of_rows

    def get_number_of_cols(self):
        return self.number_of_cols

    def __len__(self) -> int:
        return len(self.upper_left_corners)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.image_slide is not None: self.image_slide.close()
        
    def __del__(self):
        if self.image_slide is not None: self.image_slide.close()
    
    def __getitem__(self, idx : int) -> Tuple[tc.Tensor, Union[None, wsi.WholeSlideMetadata]]:
        corners, coordinates = self.get_coordinates(idx)
        patch = self.load_patch_at(corners)
        if self.transforms is not None:
            patch = tc.from_numpy(patch).to(tc.float32 if not self.half_precision else tc.float16).permute(2, 0, 1)
            patch = self.transforms(patch)
            patch = patch.contiguous()
        to_return = {'images' : patch, 'coordinates' : coordinates}
        if self.return_slide_metadata or self.return_patch_metadata:
            if self.return_slide_metadata:
                slide_metadata = {} # TODO
            else:
                slide_metadata = {}
            if self.return_patch_metadata:
                patch_metadata = {} # TODO
            else:
                slide_metadata = {}
            to_return['metadata'] = wsi.WholeSlideMetadata(patch_metadata, slide_metadata)
        return to_return