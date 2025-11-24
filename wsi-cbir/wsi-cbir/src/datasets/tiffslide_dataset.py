### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from pathlib import Path
from typing import Union, Tuple, Callable
import time

### External Imports ###
import numpy as np
import torch as tc
import tiffslide
import skimage
### Internal Imports ###
from src.datasets.bg_removal import get_bg_rm_tool
import wsi

########################


class TiffSlideDataset(wsi.WholeSlide):
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

        image_slide = tiffslide.TiffSlide(self.wsi_path)
        mask_slide = get_bg_rm_tool(self.calculate_mask_params)(image_slide) if self.calculate_mask else None
        self.mpp = (image_slide.properties['tiffslide.mpp-x']*image_slide.level_downsamples[self.resolution_level], image_slide.properties['tiffslide.mpp-y']*image_slide.level_downsamples[self.resolution_level])

        self.upper_left_corners, self.coordinates = self.calculate_upper_left_corners(image_slide, mask_slide)
        self.resolution = (image_slide.level_dimensions[self.resolution_level][1], image_slide.level_dimensions[self.resolution_level][0])

        image_slide.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.image_slide is not None: self.image_slide.close()
        
    def __del__(self):
        if self.image_slide is not None: self.image_slide.close()
           
    def calculate_upper_left_corners(self, image_slide : tiffslide.TiffSlide, mask_slide : tiffslide.TiffSlide=None):
        height = image_slide.level_dimensions[0][1] # NOTE: x, y in TiffSlide (instead of y, x in NumPy or PyTorch)
        width = image_slide.level_dimensions[0][0] # NOTE: x, y in TiffSlide (instead of y, x in NumPy or PyTorch)
        height_multiplier = 1 if self.resolution_level == 0 else int(height / image_slide.level_dimensions[self.resolution_level][1])
        width_multiplier = 1 if self.resolution_level == 0 else int(width / image_slide.level_dimensions[self.resolution_level][0])
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
        if self.image_slide is None:
            self.image_slide = tiffslide.TiffSlide(self.wsi_path)
            
    def load_patch_at(self, coordinates):
        self._ensure_image_is_open()
        patch = self.image_slide.read_region(coordinates, level=self.resolution_level, size=self.patch_size).convert('RGB')
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
    