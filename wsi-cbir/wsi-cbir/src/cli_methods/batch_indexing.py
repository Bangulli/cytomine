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
from src.datasets.dataset_mgmt import get_dataset_factory, determine_datareader
from src.retrieval.index import Index
########################
    
#------------------------------------------------ INDEXING ENTRYPOINT ------------------------------------------------#    
def calculate_embeddings(args):
    ## Handle file I/O 
    dataset = pl.Path(f"/inputs/datasets/{args.name}") if os.path.exists("/inputs") else pl.Path(f"inputs/datasets/{args.name}")
    assert dataset.exists(), f"""Path {str(dataset)} doesnt exist. Available sets are {os.listdir('/inputs/datasets') if os.path.exists("/inputs") else os.listdir('inputs/datasets')}"""
    assert dataset.is_dir(), f"Input must be a path to a BigPicture Dataset directory, {str(dataset)} is not a directory."
    embeddings = "/inputs/embeddings"/pl.Path(args.embeddings)/dataset.name if os.path.exists("/inputs") else "inputs/embeddings"/pl.Path(args.embeddings)/dataset.name
        
    ## Create or read config file
    xml_path = embeddings.parent / "embedding_config.xml"

    if not xml_path.exists():  # create if it doesn’t exist
        root = ET.Element("embeddings")
        
        # --- Config section ---
        config = ET.SubElement(root, "config", {
            "encoder": args.encoder,
            "level": str(args.level),
            "full_precision": str(args.full_precision),
            "remove_bg": str(args.remove_bg)
        })

        # --- Datasets section ---
        datasets_el = ET.SubElement(root, "datasets")
        ET.SubElement(datasets_el, "dataset", {
            "name": dataset.name,
            "fully_processed": "False"
        })

        # --- Write XML to file ---
        tree = ET.ElementTree(root)
        ET.indent(tree, space="\t", level=0)  
        os.makedirs(embeddings, exist_ok=True)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    else:  # override if it exists
        print("= Found embedding_config.xml in parent directory, overriding any arguments to ensure homogenity")

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # --- Override args from <config> ---
        config = root.find("config")
        if config is not None:
            for key, value in config.attrib.items():
                val = value
                setattr(args, key, val)
        
        ## set booleans
        args.full_precision = args.full_precision.lower()=='true' if type(args.full_precision) != bool else args.full_precision


    # --- Handle datasets ---
    datasets_el = root.find("datasets")
    if datasets_el is None:
        datasets_el = ET.SubElement(root, "datasets")

    all_sets = {ds.get("name"): ds.get("fully_processed") for ds in datasets_el.findall("dataset")}

    if dataset.name in all_sets:
        if all_sets[dataset.name] == "True":
            result = {
                "status": "Finished",
                "info": f"Dataset {dataset.name} is listed as fully processed in embeddings_config.xml, skipping."
            }
            print("--FINISHED--", json.dumps(result))
            if hasattr(args, 'return_result'):
                return result
            else:
                return 0
    else:
        ET.SubElement(datasets_el, "dataset", {
            "name": dataset.name,
            "fully_processed": "False"
        })
        os.makedirs(embeddings, exist_ok=True)

    # --- Write updated XML ---
    ET.indent(tree, space="\t", level=0)
    tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            
    ## Handle Datatype
    factory = get_dataset_factory(determine_datareader(dataset))
        
    ## Handle Encoder
    device = get_least_used_gpu()
    patch_encoder, slide_encoder, transforms, patch_size = get_encoder(args.encoder)
    patch_encoder = patch_encoder.to(device)
    slide_encoder = slide_encoder.to(device)
    from src.networks.patch_encoders import patch_encoder as pe
    from src.networks.slide_encoders import slide_encoder as se
    patch_encoder = pe.PatchEncoder(patch_encoder)
    slide_encoder = se.SlideEncoder(slide_encoder)
    
    slides = os.listdir(dataset/"IMAGES")
    count = len(slides)
    if not (embeddings/"indexed.xml").is_file():
        print(f"= Found {count} wsi files in {str(dataset)}, starting encoding")
    else:
        print(f"= Found {count} wsi files in {str(dataset)}, but dataset is partially processed, checking remaining")
        resume = minidom.parse(str(embeddings/"indexed.xml"))
        processed_slides = [f.getAttribute('wsi') for f  in resume.getElementsByTagName("sample")]
        slides = [f for f in slides if f not in processed_slides]
        count = len(slides)
        print(f"= There are {count} wsi files remaining unprocessed in {str(dataset)}, resuming encoding")
        
    global_start = time.time()
    idx = 0
    width = len(str(count))
    xml_path = embeddings / "indexed.xml"
    avg_time_per_wsi = 0
    avg_time_per_gigapixel = 0
    
    precision = tc.float32 if args.full_precision else tc.float16
    with tc.autocast('cuda', precision), tc.no_grad():
        ## WSI processing loop
        for slide in slides:
                tc.cuda.empty_cache()
                idx += 1
                print(f"--------------------------({idx:-{width}d}/{count})--------------------------")
                wsi_path = dataset/"IMAGES"/slide
                embedding, time_elapsed, sGP = calculate_embedding(precision, factory, device, wsi_path, int(args.level), int(patch_size), int(patch_size), args.remove_bg, transforms, patch_encoder, slide_encoder)
                avg_time_per_wsi += time_elapsed
                avg_time_per_gigapixel += sGP
                emb_pth = embeddings/f"{wsi_path.stem}_embedding.pth"
                embedding.save_embedding(emb_pth)
                print(f"""== Saved embedding to {emb_pth}""")
                
                if not xml_path.exists():
                    root = ET.Element("dataset", {
                        "total": str(count),
                        "name": dataset.name
                    })
                else:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                ET.SubElement(root, "sample", {
                    "wsi": str(wsi_path.name),
                    "mask": str(None),
                    "embedding": f"{wsi_path.stem}_embedding.pth"
                })
                tree = ET.ElementTree(root)
                ET.indent(tree, space="\t", level=0) 
                tree.write(xml_path, encoding="utf-8", xml_declaration=True)
                
        
    ## keep in memory and only write in the end to avoid constantly parsing and rewriting the xml on disk
    min, s = divmod(time.time()-global_start, 60)
    h, min = divmod(min, 60) 
    print(f"= Indexing {count} files took {h:.0f}h {min:.0f}min {s:.2f}s")
    print(f"= {args.encoder} takes on avg {(avg_time_per_wsi/count):.2f} s/Img and {(avg_time_per_gigapixel/count):.2f} s/GP")
    
    ## check if all image files have been embedded, if so change the flag in the embeddings_config.xml
    tree = ET.parse(xml_path)
    root = tree.getroot()
    processed = [sample.get("wsi") for sample in root.findall("sample")]
    if len(processed) == int(root.get("total", "0")):
        print(f"Successfully processed all WSI in {str(dataset.name)}")
        ## add to indexer
        index = Index(embeddings.parent, DIMS[args.encoder]) if not (embeddings.parent/'index.faiss').exists() else Index(embeddings.parent).load()
        id0, id1 = index.add_dir(embeddings)
        ## write to xml
        xml_path = embeddings.parent / "embedding_config.xml"
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for ds in root.findall("./datasets/dataset"):
            if ds.get("name") == str(dataset.name):
                ds.set("fully_processed", "True")
                ds.set("indexID0", str(id0))
                ds.set("indexID1", str(id1))
        ET.indent(tree, space="\t", level=0)
        tree.write(xml_path, encoding="utf-8", xml_declaration=True)
        
    
    ## send response
    result = {
        'status': 'Finished',
        'info': f"Indexing {count} files took {h:.0f}h {min:.0f}min {s:.2f}s"
    }
    print("--FINISHED--", json.dumps(result))
    if hasattr(args, 'return_result'):
        return result
    else:
        return 0

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
    
