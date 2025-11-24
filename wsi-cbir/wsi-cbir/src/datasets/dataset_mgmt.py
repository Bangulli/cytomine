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
from src.datasets import factories 
from src.utils.switch_case import SwitchCase
########################

def get_dataset_factory(key):
    ## Load models
    with SwitchCase(key) as switch:
        if switch.case("wsidicom"): 
            print("@dataset-mgmt: Parsing input as containing dicom directories with wsidicom")
            factory = factories.make_wsidicom_dataset
            
        elif switch.case("openslide"): 
            print("@dataset-mgmt: Parsing input with openslide")
            factory = factories.make_openslide_dataset
            
        elif switch.case("tiffslide"): 
            print("@dataset-mgmt: Parsing input with tiffslide")
            factory = factories.make_tiffslide_dataset
            
    return factory

def determine_datareader(path):
    if all([(path/'IMAGES'/f).is_dir() for f in os.listdir(path/'IMAGES')]):
        return "wsidicom"
    elif all([(path/'IMAGES'/f).is_file() for f in os.listdir(path/'IMAGES')]):
        return "openslide"
    else: raise RuntimeError(f"Could not determine datareader as the contents of {str(path/'IMAGES')} is not homogeneous")
    
def determine_datareader_for_file(path):
    if path.is_dir():
        return "wsidicom"
    elif path.is_file():
        return "openslide"
    else: raise RuntimeError(f"Could not determine datareader as the contents of {str(path)} is not homogeneous")
    
# def get_dataset_factory(key):
#     # this could literally just be opt[key] lol but then the connection with the indexing -h option printout would break
#     return SwitchCase.switch_case_function(
#         key=key,
#         options= {
#         'wsidicom': factories.make_wsidicom_dataset,
#         'openslide': factories.make_openslide_dataset,
#         'tiffslide': factories.make_tiffslide_dataset
#         })