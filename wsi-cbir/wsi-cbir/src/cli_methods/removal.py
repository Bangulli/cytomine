### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib as pl
import warnings
warnings.filterwarnings('ignore')
from xml.dom import minidom
import time
import xml.etree.ElementTree as ET
import json
### External Imports ###
import torch as tc
### Internal Imports ###
from src.inference import inference
from src.datasets.wsi import WholeSlideEmbedding
from src.utils.hardware_mgmt import get_least_used_gpu
from src.networks.encoder_mgmt import get_encoder, DIMS
from src.datasets.dataset_mgmt import get_dataset_factory, determine_datareader_for_file
from src.retrieval.index import Index
from src.config import CYTOMINE_CONFIG
########################
    
#------------------------------------------------ INDEXING ENTRYPOINT ------------------------------------------------#    
def remove_embedding_for_image(path, filename, image_id):
    ## Handle file I/O 
    image_path = pl.Path('/images')/path
    image_filename = filename
    image_id = image_id
    embeddings = pl.Path(CYTOMINE_CONFIG['embeddings'])

    xml_path = embeddings / "indexed.xml"
    emb_pth = embeddings/f"{image_filename}_embedding.pth"
    
    ## remove from index
    index = Index(embeddings).load()
    index.rm([image_filename])
    os.remove(emb_pth)
    
    ## remove from xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_id_str = str(image_id)
    for sample in list(root.findall("sample")):  
        if sample.get("id") == image_id_str:
            root.remove(sample)
    ET.indent(tree, space="\t", level=0) 
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    ## send response
    print(f"= Removed image {image_filename} from index")
    result = {
        'status': 'Finished',
        'info': f"Removed image {image_filename} from index"
    }
    return result

