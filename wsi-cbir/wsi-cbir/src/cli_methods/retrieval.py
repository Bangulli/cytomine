### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import pathlib as pl
import xml.etree.ElementTree as ET
### External Imports ###
import json
import torch as tc
import numpy as np
### Internal Imports ###
from src.retrieval.index import Index
from src.cli_methods.indexing import calculate_embedding
from src.datasets.wsi import WholeSlideEmbedding
from src.utils import metadata_filtration
from src.utils.hardware_mgmt import get_least_used_gpu
from src.networks.encoder_mgmt import get_encoder, DIMS
from src.datasets.dataset_mgmt import get_dataset_factory, determine_datareader_for_file
########################
#------------------------------------------------ UTIL ------------------------------------------------#  
def get_dataset_ids(embeddings, dataset):
    """Returns a list of IDs for a given dataset

    Args:
        embeddings (Path): Path to the embeddings directory
        dataset (str): Name of the dataset subdirectory

    Returns:
        list: List of ID strings
    """
    xml_path = embeddings / "embedding_config.xml"
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for ds in root.findall("./datasets/dataset"):
        if ds.get("name") == str(dataset) and ds.get("fully_processed") == "True":
            print(f"== Parsing {dataset}")
            xml_path = embeddings / dataset / "indexed.xml"
            tree = ET.parse(xml_path)
            root = tree.getroot()
            return [f"""{dataset}/{name.get("wsi")}""" for name in root.findall("sample")]
    raise RuntimeError(f"Requested dataset {dataset} is not fully encoded")
        
#------------------------------------------------ RETRIEVAL ENTRYPOINT ------------------------------------------------#        
def find_k_similar(args):
    if args.query is None and args.metadata is None:
        raise RuntimeError(f"Retrieval received neither query nor metadata to perform any search")
    ## Handle dataset path
    embeddings = "/inputs/embeddings"/pl.Path(args.embeddings) if os.path.exists('/inputs') else "inputs/embeddings"/pl.Path(args.embeddings)
    if not embeddings.is_dir():
        raise ValueError(f"""Embeddings path must lead to a valid directory, got {args.embeddings} available is {os.listdir("/inputs/embeddings") if os.path.exists('/inputs') else os.listdir("inputs/embeddings")}""")
    elif not (embeddings/"embedding_config.xml").exists():
        raise ValueError("Embeddings directory must contain embedding_config.xml")
        
    ## Parse indexed.xml
    xml = ET.parse(embeddings/"embedding_config.xml")
    emb_root = xml.getroot()
    
    ## read config
    config = emb_root.findall('./config')[0].attrib
    encoder = config['encoder']
    level = int(config['level'])
    full_precision = config['full_precision'] == "True" # workaround cause direct casting of literal to bool is not supported
    remove_bg = config['remove_bg']
        
    ## Handle query file type
    if args.query is not None:
        query = pl.Path(args.query)
        if query.name.endswith(".pth"): # skip encoding step
            print("= Treating query file as pre-encoded embedding")
            query_embedding  = WholeSlideEmbedding().load_embedding(path=query).squeeze().numpy()
            print("= Embedding loaded successfully!")
            
        else: # run inference on query image
            print("= Treating query image as WSI")
            
            ## Handle encoder I/O
            print(f"= Loading model {encoder}")
            device = 'cpu' if args.cpu else get_least_used_gpu()
            patch_encoder, slide_encoder, transforms, patch_size = get_encoder(encoder)
            patch_encoder = patch_encoder.to(device)
            slide_encoder = slide_encoder.to(device)
            from src.networks.patch_encoders import patch_encoder as pe
            from src.networks.slide_encoders import slide_encoder as se
            patch_encoder = pe.PatchEncoder(patch_encoder)
            slide_encoder = se.SlideEncoder(slide_encoder)
            
            ## Handle Dataloading
            factory = get_dataset_factory(determine_datareader_for_file(query))
            
            ## Obtain embedding
            precision = tc.float32 if full_precision else tc.float16
            query_embedding, _, _ = calculate_embedding(precision, factory, device, query, level, patch_size, patch_size, remove_bg, transforms, patch_encoder, slide_encoder)
            query_embedding = query_embedding.numpy()
    
    ## Handle subset selection by dataset selection and metadata filtration  
    subset = []
    if args.datasets:
        print(f'= Selecting subset by dataset name')
        for name in args.datasets:
            subset += get_dataset_ids(embeddings, name)
            
    if args.query is not None:
        # Perform search
        index = Index(embeddings).load()
        
        if args.metadata:
            print(f'= Filtering subset by metadata')
            if type(args.metadata) is dict:
                subset = index.filter_metadata(args.metadata, subset if subset else None)
            elif type(args.metadata) is str:
                with open(args.metadata, 'r') as file:
                    metadata_filter = json.load(file)
                subset = index.filter_metadata(metadata_filter, subset if subset else None)
            else:
                raise RuntimeError('Metadata filter must be either dict or path to json file')
    
        if subset:
            print(f'= Performing search in subset of size {len(subset)}')
            best_imgs, best_sims = index.search_subset(np.expand_dims(query_embedding, 0), int(args.k_best), subset)
        else:
            print(f"= Searching entire database of size {index.ntotal}")
            best_imgs, best_sims = index.search(np.expand_dims(query_embedding, 0), int(args.k_best))

        result = {
            "query": args.query,
            "metadata-filter": args.metadata,
            "target-dataset": args.datasets,
            "embedding_database": args.embeddings,
            "similarities": list(zip(best_imgs, best_sims))
        }
        if args.save is not None:
            with open(args.save if args.save.endswith('.json') else args.save+'.json', 'w') as file:
                print(f"""== Writing results to file {args.save if args.save.endswith('.json') else args.save+'.json'}""")
                json.dump(result, file, indent=4)
        print("--FINISHED--", json.dumps(result))
        if hasattr(args, 'return_result'):
            return result
        else:
            return 0
    else:
        print(f"= Received no query, will fetch only metadata based results")
        
        index = Index(embeddings).load()
        
        if args.metadata:
            print(f'= Filtering subset by metadata')
            subset = index.filter_metadata(args.metadata, subset if subset else None)
        else: raise RuntimeError('Didnt receive metadata filter or query')
            
        result = {
            "query": args.query,
            "metadata-filter": args.metadata,
            "target-dataset": args.datasets,
            "embedding_database": args.embeddings,
            "filtered": subset
        }
        if args.save is not None:
            with open(args.save if args.save.endswith('.json') else args.save+'.json', 'w') as file:
                print(f"""== Writing results to file {args.save if args.save.endswith('.json') else args.save+'.json'}""")
                json.dump(result, file, indent=4)
        print("--FINISHED--", json.dumps(result))
        if hasattr(args, 'return_result'):
            return result
        else:
            return 0
    
        
