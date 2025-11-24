### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import warnings
warnings.filterwarnings('ignore')
from typing import Any
### External Imports ###

### Internal Imports ###
from src.utils.switch_case import SwitchCase
########################
DIMS = {
    'PGP': 768,
    'ProvGigaPath': 768,
    'CHIEF': 768,
    'TITAN': 768,
    'PRISM': 1280,
}
  
def get_encoder(key):
    ## Load models
    with SwitchCase(key) as switch:
        if switch.case(["ProvGigaPath", "PGP"]):
            patch_model_checkpoint_path = "./model_weights/provgigapath_patch_raw.pth"
            slide_model_checkpoint_path = "./model_weights/provgigapath_slide.pth"
            from src.networks.patch_encoders import provgigapath as provgigapath_patch
            from src.networks.slide_encoders import provgigapath as provgigapath_slide
            patch_encoder = provgigapath_patch.ProvGigaPath_Patch(model_checkpoint_path=patch_model_checkpoint_path).eval()
            slide_encoder = provgigapath_slide.ProvGigaPath_Slide(model_checkpoint_path=slide_model_checkpoint_path).eval()
            transforms = provgigapath_patch.get_transform()
            patch_size = 518
            print(f"@encoder-mgmt: ProvGigaPath loaded successfully.")
            
        elif switch.case("CHIEF"):
            patch_model_checkpoint_path = "./model_weights/CHIEF_patch.pth"
            slide_model_checkpoint_path = "./model_weights/CHIEF_slide.pth"
            from src.networks.patch_encoders import chief as chief_patch
            from src.networks.slide_encoders import chief as chief_slide
            patch_encoder = chief_patch.CHIEF_Patch(model_checkpoint_path=patch_model_checkpoint_path).eval()
            slide_encoder = chief_slide.CHIEF_Slide(model_checkpoint_path=slide_model_checkpoint_path).eval()
            transforms = chief_patch.get_transform()
            patch_size = 224
            print(f"@encoder-mgmt: CHIEF loaded successfully.")
            
        elif switch.case("TITAN"):
            patch_model_checkpoint_path = "./model_weights/TITAN_patch.bin"
            slide_model_checkpoint_path = "./model_weights/TITAN_slide.safetensors"
            from src.networks.patch_encoders import titan as titan_patch
            from src.networks.slide_encoders import titan as titan_slide
            patch_encoder = titan_patch.TITAN_Patch(model_checkpoint_path=patch_model_checkpoint_path).eval()
            slide_encoder = titan_slide.TITAN_Slide(model_checkpoint_path=slide_model_checkpoint_path).eval()
            transforms = patch_encoder.get_transforms()
            patch_size = 448
            print(f"@encoder-mgmt: TITAN loaded successfully")
            
        elif switch.case("PRISM"):
            patch_model_checkpoint_path = "./model_weights/PRISM_patch.bin"
            slide_model_checkpoint_path = "./model_weights/PRISM_slide.pth"
            from src.networks.patch_encoders import prism as prism_patch
            from src.networks.slide_encoders import prism as prism_slide
            patch_encoder = prism_patch.PRISM_Patch(model_checkpoint_path=patch_model_checkpoint_path).eval()
            slide_encoder = prism_slide.PRISM_Slide(model_checkpoint_path=slide_model_checkpoint_path).eval()
            transforms = patch_encoder.get_transforms()
            patch_size = 224
            print(f"@encoder-mgmt: PRISM loaded successfully")
            
    return patch_encoder, slide_encoder, transforms, patch_size


