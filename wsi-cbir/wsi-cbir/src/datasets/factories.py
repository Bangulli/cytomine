import openslide_dataset
import wsidicom_dataset
import tiffslide_dataset
import wsi
from typing import Union, Callable
from pathlib import Path

def make_openslide_dataset(wsi_path : Union[str, Path],
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
        half_precision : bool = False,):
    ds = ds = openslide_dataset.OpenSlideDataset(
       wsi_path,
       mask_path,
       metadata_path,
       resolution_level,
       patch_size,
       patch_stride, 
       return_patch_metadata,
       return_slide_metadata,
       calculate_mask,
       calculate_mask_params,
       transforms,
       half_precision
    )
    return ds

def make_tiffslide_dataset(wsi_path : Union[str, Path],
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
        half_precision : bool = False,):
    ds = ds = tiffslide_dataset.TiffSlideDataset(
       wsi_path,
       mask_path,
       metadata_path,
       resolution_level,
       patch_size,
       patch_stride, 
       return_patch_metadata,
       return_slide_metadata,
       calculate_mask,
       calculate_mask_params,
       transforms,
       half_precision
    )
    return ds

def make_wsidicom_dataset(wsi_path : Union[str, Path],
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
        half_precision : bool = False,):
    ds = ds = wsidicom_dataset.WsiDicomDataset(
       wsi_path,
       mask_path,
       metadata_path,
       resolution_level,
       patch_size,
       patch_stride, 
       return_patch_metadata,
       return_slide_metadata,
       calculate_mask,
       calculate_mask_params,
       transforms,
       half_precision
    )
    return ds