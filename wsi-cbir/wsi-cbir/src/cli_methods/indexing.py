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
def calculate_embedding_for_image(path, filename, image_id):
    ## Handle file I/O 
    image_path = pl.Path('/images')/path
    image_filename = filename
    image_id = image_id
    embeddings = pl.Path(CYTOMINE_CONFIG['embeddings'])
     
    ## Handle Datatype
    factory = get_dataset_factory(determine_datareader_for_file(image_path))
        
    ## Handle Encoder
    device = get_least_used_gpu()
    patch_encoder, slide_encoder, transforms, patch_size = get_encoder(CYTOMINE_CONFIG['encoder'])
    patch_encoder = patch_encoder.to(device)
    slide_encoder = slide_encoder.to(device)
    from src.networks.patch_encoders import patch_encoder as pe
    from src.networks.slide_encoders import slide_encoder as se
    patch_encoder = pe.PatchEncoder(patch_encoder)
    slide_encoder = se.SlideEncoder(slide_encoder)
    
    ## initialize variables for embedding loop
    global_start = time.time()
    xml_path = embeddings / "indexed.xml"
    precision = tc.float32 if CYTOMINE_CONFIG['full_precision'] else tc.float16

    #-COMPUTE EMBEDDING--------------------    
    with tc.autocast(device.split(':')[0], precision), tc.no_grad():
        tc.cuda.empty_cache()
        embedding, _, sGP = calculate_embedding(precision, factory, device, image_path, CYTOMINE_CONFIG['level'], int(patch_size), int(patch_size), CYTOMINE_CONFIG['remove_bg'], transforms, patch_encoder, slide_encoder)
        emb_pth = embeddings/f"{image_id}_embedding.pth"
        embedding.save_embedding(emb_pth)
        print(f"""== Saved embedding to {emb_pth}""")
        
        if not xml_path.exists():
            root = ET.Element("dataset")
        else:
            tree = ET.parse(xml_path)
            root = tree.getroot()

        ET.SubElement(root, "sample", {
            "path": str(image_path),
            "name": str(image_filename),
            "id": str(image_id),
            "embedding": emb_pth.name,
        })
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0) 
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
    #-COMPUTE EMBEDDING--------------------   
        
    ## keep in memory and only write in the end to avoid constantly parsing and rewriting the xml on disk
    min, s = divmod(time.time()-global_start, 60)
    print(f"= Indexing image {image_filename} took {min:.0f}min {s:.2f}s; that is {sGP}s/Gigapixel")
    
    ## add to index
    index = Index(embeddings, DIMS[CYTOMINE_CONFIG['encoder']]) if not (embeddings/'index.faiss').exists() else Index(embeddings).load()
    _, _ = index.add(embedding.unsqueeze(0), [image_id], False)

    ## send response
    result = {
        'status': 'Finished',
        'info': f"Indexing image {image_filename} took {min:.0f}min {s:.2f}s"
    }
    print("--FINISHED--", json.dumps(result))
    return result

def calculate_embedding(precision, factory, device, image, level, patch_size, patch_stride, remove_bg, transforms, patch_encoder, slide_encoder):
    with tc.autocast('cuda' if 'cuda' in device else 'cpu', precision):
        start = time.time()
        ## Encode query image
        wsi = factory(
                wsi_path = image,
                mask_path = None,
                metadata_path = None,
                resolution_level = level,
                patch_size = (patch_size, patch_size),
                patch_stride = (patch_stride, patch_stride),
                return_patch_metadata = False,
                return_slide_metadata = False,
                calculate_mask = not remove_bg.lower()=='false',
                calculate_mask_params = remove_bg, # ignorded if above is false
                transforms = transforms,  
                half_precision = not precision == tc.float32
            )
        print(f"== Calculating embeddings for {str(image)}")
        gigapixels = (wsi.resolution[0]*wsi.resolution[1])/1e9
        print(f"== Handling image with Level={wsi.resolution_level}, {wsi.mpp[0]:.2f} x {wsi.mpp[1]:.2f} µm/pixel, {wsi.resolution[0]} x {wsi.resolution[1]} pixels -> {gigapixels:.2f} GP")
                    
        query_embedding : WholeSlideEmbedding = inference.calculate_slide_level_embedding(
            wsi,
            patch_encoder,
            slide_encoder,
            batch_size=64,
            num_workers=64,
            device=device,
            echo=True,
        ).type(tc.float16)
        time_elapsed=time.time()-start
        sGP = time_elapsed/gigapixels
        print(f"== Calculating embeddings took {time_elapsed:.2f} seconds, that is {sGP:.2f} s/GP")
        return query_embedding.to("cpu").squeeze(), time_elapsed, sGP
    
