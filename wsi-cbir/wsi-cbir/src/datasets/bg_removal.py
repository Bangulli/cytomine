### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import logging ## avoid weird wsidicom logspam: WARNING:root:Orientation [-1.0, 0.0, 0.0, 0.0, -1.0, 0.0] is not orthogonal with equal lengths with column rotated 90 deg from row
logging.getLogger().setLevel(logging.ERROR)
from typing import Any
### External Imports ###
import numpy as np
import skimage
import wsidicom
import openslide
import PIL
import tiffslide
import cv2
### Internal Imports ###
from src.utils.switch_case import SwitchCase
########################

class Otsu:
    def __init__(self, wsi):
        self.wsi = wsi
        thumbnail = self._get_thumbnail()
        w, h = thumbnail.size
        thumbnail = thumbnail.crop((w*0.1, h*0.1, w*0.9, h*0.9)) ## Crop edges to avoid bad thresholding
        self.gs_threshold = skimage.filters.threshold_otsu(np.array(thumbnail)) 
        print(f"==== Otsu threshold is {self.gs_threshold}")
        
    def _get_thumbnail(self, size=512):
        with SwitchCase(type(self.wsi)) as switch:
            if switch.case(wsidicom.WsiDicom):
                return self.wsi.read_thumbnail((size, size)).convert("L")
            elif switch.case(tiffslide.TiffSlide):
                return self.wsi.get_thumbnail((size, size)).convert("L") 
            elif switch.case(openslide.OpenSlide):
                return self.wsi.get_thumbnail((size, size)).convert("L") 
    
    def read_region(self, coords, level, patch_size):
        mask_patch = self.wsi.read_region(coords, level, patch_size).convert("L")
        mask_patch = np.array(mask_patch)<self.gs_threshold # binarize: bg is bright, fg is dark
        return mask_patch
    
    def close(self): # NOTE: usually doesnt need to be called, as the underlying wsi is closed at the end of dataset.__init__ already
        self.wsi.close()
        
    def get_segmented_thumbnail_PIL(self):
        thmbnl = self._get_thumbnail()
        thmbnl = np.array(thmbnl)<self.gs_threshold
        thmbnl = PIL.Image.fromarray(thmbnl)
        return thmbnl

class DilatedOtsu:
    def __init__(self, wsi, se_rad_microns=64):
        self.wsi = wsi
        thumbnail = self._get_thumbnail()
        w, h = thumbnail.size
        thumbnail = thumbnail.crop((w*0.1, h*0.1, w*0.9, h*0.9)) ## Crop edges to avoid bad thresholding
        self.gs_threshold = skimage.filters.threshold_otsu(np.array(thumbnail)) 
        print(f"==== Otsu threshold is {self.gs_threshold}")
        self.se_rad_microns = se_rad_microns
        
        # None inits
        self.seg_thmbnl = None
        self.multiplier = None
        
    def _get_thumbnail(self, size=512):
        with SwitchCase(type(self.wsi)) as switch:
            if switch.case(wsidicom.WsiDicom):
                return self.wsi.read_thumbnail((size, size)).convert("L")
            elif switch.case(tiffslide.TiffSlide):
                return self.wsi.get_thumbnail((size, size)).convert("L") 
            elif switch.case(openslide.OpenSlide):
                return self.wsi.get_thumbnail((size, size)).convert("L") 
            
    def _get_level_mpp(self, level):
        with SwitchCase(type(self.wsi)) as switch:
            if switch.case(wsidicom.WsiDicom):
                mpp = self.wsi.levels[level].mpp.height
                return mpp
            elif switch.case(openslide.OpenSlide):
                try:
                    mpp = (self.wsi.properties[openslide.PROPERTY_NAME_MPP_X]*self.wsi.level_downsamples[level], self.wsi.properties[openslide.PROPERTY_NAME_MPP_Y]*self.wsi.level_downsamples[level])
                except:
                    mpp = (0.5, 0.5) # Default X20
                return mpp[0]
            elif switch.case(tiffslide.TiffSlide):
                mpp = (self.wsi.properties['tiffslide.mpp-x']*self.wsi.level_downsamples[level], self.wsi.properties['tiffslide.mpp-y']*self.wsi.level_downsamples[level])
                return mpp[0]
    
    # def read_region_slow(self, coords, level, patch_size):
    #     if self.se is None: self.se = self._get_se(level)
    #     mask_patch = self.wsi.read_region(coords, level, patch_size).convert("L")
    #     mask_patch = np.array(mask_patch)<self.gs_threshold # binarize: bg is bright, fg is dark
    #     mask_patch = cv2.dilate(mask_patch.astype(np.uint8), self.se, iterations=1)#skimage.morphology.dilation(mask_patch, self.se)
    #     return mask_patch
    
    def read_region(self, coords, level, patch_size): ## passed coords are w, h
        if self.seg_thmbnl is None: self.seg_thmbnl = self.get_segmented_thumbnail(level, 5000)
        if self.multiplier is None: self.multiplier = np.array(self._get_region_to_thmbnl_converter(level)); print(f"=== Patch size scaled to {np.round(patch_size*self.multiplier).astype(int)} in thumbnail of size {self.seg_thmbnl.shape}")
        coords_in_thmbnl = np.round(coords*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.round(patch_size*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.maximum(patch_size_in_thmbnl, [1, 1])
        return self.seg_thmbnl[coords_in_thmbnl[1]:coords_in_thmbnl[1]+patch_size_in_thmbnl[1], coords_in_thmbnl[0]:coords_in_thmbnl[0]+patch_size_in_thmbnl[0]] ## NOTE flipped coords, because numpy expects height, width
    
    def _get_region_to_thmbnl_converter(self, level): # returns width, height
        thmbnl_shape = self.seg_thmbnl.shape ## height, width
        wsi_at_level_shape = self._get_resolution(level)
        return thmbnl_shape[1]/wsi_at_level_shape[0], thmbnl_shape[0]/wsi_at_level_shape[1]
        
    def _get_resolution(self, level): # returns width, height
        with SwitchCase(type(self.wsi)) as switch:
            if switch.case(wsidicom.WsiDicom):
                return (self.wsi.levels[level].size.width, self.wsi.levels[level].size.height)
            elif switch.case(openslide.OpenSlide):
                return (self.wsi.level_dimensions[level][0], self.wsi.level_dimensions[level][1])
            elif switch.case(tiffslide.TiffSlide):
                return (self.wsi.level_dimensions[level][0], self.wsi.level_dimensions[level][1])
        
    def close(self): # NOTE: usually doesnt need to be called, as the underlying wsi is closed at the end of dataset.__init__ already
        self.wsi.close()
        
    def get_segmented_thumbnail(self, level=0, size=512):
        thmbnl = self._get_thumbnail(size=size)
        thmbnl = np.array(thmbnl)<self.gs_threshold
        mpp = self._get_level_mpp(level)
        dims = self._get_resolution(level) # w, h
        thmbnl_dims = thmbnl.shape # h, w
        x = dims[0]/thmbnl_dims[1]
        thmbnl_se = skimage.morphology.disk(round(self.se_rad_microns/(mpp*x)))
        #thmbnl = skimage.morphology.dilation(thmbnl, thmbnl_se)
        thmbnl = cv2.dilate(thmbnl.astype(np.uint8), thmbnl_se, iterations=1).astype(bool)
        return thmbnl
    
    def get_segmented_thumbnail_PIL(self, level=0, size=5000):
        return PIL.Image.fromarray(self.get_segmented_thumbnail(level, size))
    
class CleanedOtsu:
    def __init__(self, wsi, se_rad_microns=64):
        self.wsi = wsi
        thumbnail = self._get_thumbnail()
        w, h = thumbnail.size
        thumbnail = thumbnail.crop((w*0.1, h*0.1, w*0.9, h*0.9)) ## Crop edges to avoid bad thresholding
        self.gs_threshold = skimage.filters.threshold_otsu(np.array(thumbnail)) 
        print(f"==== Otsu threshold is {self.gs_threshold}")
        self.se_rad_microns = se_rad_microns
        
        # None inits
        self.seg_thmbnl = None
        self.multiplier = None
        
    def _get_thumbnail(self, size=512):
        with SwitchCase(type(self.wsi)) as switch:
            if switch.case(wsidicom.WsiDicom):
                return self.wsi.read_thumbnail((size, size)).convert("L")
            elif switch.case(tiffslide.TiffSlide):
                return self.wsi.get_thumbnail((size, size)).convert("L") 
            elif switch.case(openslide.OpenSlide):
                return self.wsi.get_thumbnail((size, size)).convert("L") 
            
    def _get_level_mpp(self, level):
        with SwitchCase(type(self.wsi)) as switch:
            if switch.case(wsidicom.WsiDicom):
                mpp = self.wsi.levels[level].mpp.height
                return mpp
            elif switch.case(openslide.OpenSlide):
                try:
                    mpp = (self.wsi.properties[openslide.PROPERTY_NAME_MPP_X]*self.wsi.level_downsamples[level], self.wsi.properties[openslide.PROPERTY_NAME_MPP_Y]*self.wsi.level_downsamples[level])
                except:
                    mpp = (0.5, 0.5) # Default X20
                return mpp[0]
            elif switch.case(tiffslide.TiffSlide):
                mpp = (self.wsi.properties['tiffslide.mpp-x']*self.wsi.level_downsamples[level], self.wsi.properties['tiffslide.mpp-y']*self.wsi.level_downsamples[level])
                return mpp[0]
    
    # def read_region_slow(self, coords, level, patch_size):
    #     if self.se is None: self.se = self._get_se(level)
    #     mask_patch = self.wsi.read_region(coords, level, patch_size).convert("L")
    #     mask_patch = np.array(mask_patch)<self.gs_threshold # binarize: bg is bright, fg is dark
    #     mask_patch = cv2.dilate(mask_patch.astype(np.uint8), self.se, iterations=1)#skimage.morphology.dilation(mask_patch, self.se)
    #     return mask_patch
    
    def read_region(self, coords, level, patch_size): ## passed coords are w, h
        if self.seg_thmbnl is None: self.seg_thmbnl = self.get_segmented_thumbnail(level, 5000)
        if self.multiplier is None: self.multiplier = np.array(self._get_region_to_thmbnl_converter(level)); print(f"=== Patch size scaled to {np.round(patch_size*self.multiplier).astype(int)} in thumbnail of size {self.seg_thmbnl.shape}")
        coords_in_thmbnl = np.round(coords*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.round(patch_size*self.multiplier).astype(int)
        patch_size_in_thmbnl = np.maximum(patch_size_in_thmbnl, [1, 1])
        return self.seg_thmbnl[coords_in_thmbnl[1]:coords_in_thmbnl[1]+patch_size_in_thmbnl[1], coords_in_thmbnl[0]:coords_in_thmbnl[0]+patch_size_in_thmbnl[0]] ## NOTE flipped coords, because numpy expects height, width
    
    def _get_region_to_thmbnl_converter(self, level): # returns width, height
        thmbnl_shape = self.seg_thmbnl.shape ## height, width
        wsi_at_level_shape = self._get_resolution(level)
        return thmbnl_shape[1]/wsi_at_level_shape[0], thmbnl_shape[0]/wsi_at_level_shape[1]
        
    def _get_resolution(self, level): # returns width, height
        with SwitchCase(type(self.wsi)) as switch:
            if switch.case(wsidicom.WsiDicom):
                return (self.wsi.levels[level].size.width, self.wsi.levels[level].size.height)
            elif switch.case(openslide.OpenSlide):
                return (self.wsi.level_dimensions[level][0], self.wsi.level_dimensions[level][1])
            elif switch.case(tiffslide.TiffSlide):
                return (self.wsi.level_dimensions[level][0], self.wsi.level_dimensions[level][1])
        
    def close(self): # NOTE: usually doesnt need to be called, as the underlying wsi is closed at the end of dataset.__init__ already
        self.wsi.close()
        
    def get_segmented_thumbnail(self, level=0, size=512):
        thmbnl = self._get_thumbnail(size=size)
        thmbnl = np.array(thmbnl)<self.gs_threshold
        mpp = self._get_level_mpp(level)
        dims = self._get_resolution(level) # w, h
        thmbnl_dims = thmbnl.shape # h, w
        x = dims[0]/thmbnl_dims[1]
        thmbnl_se = skimage.morphology.disk(round(self.se_rad_microns/(mpp*x)))
        thmbnl = cv2.morphologyEx(thmbnl.astype(np.uint8), cv2.MORPH_CLOSE, thmbnl_se).astype(bool)
        thmbnl = cv2.morphologyEx(thmbnl.astype(np.uint8), cv2.MORPH_OPEN, thmbnl_se).astype(bool)
        return thmbnl
    
    def get_segmented_thumbnail_PIL(self, level=0, size=5000):
        return PIL.Image.fromarray(self.get_segmented_thumbnail(level, size))
    
def get_bg_rm_tool(method): # NOTE: to extend
    with SwitchCase(method) as switch:
        if switch.case('otsu'):
            return Otsu
        elif switch.case(['dilated-otsu', 'dilated_otsu']):
            return DilatedOtsu
        elif switch.case(['cleaned-otsu', 'cleaned_otsu']):
            return CleanedOtsu
    