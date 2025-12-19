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
from src.config import CYTOMINE_CONFIG
########################
        
#------------------------------------------------ RETRIEVAL ENTRYPOINT ------------------------------------------------#        
def find_k_similar(index: Index, query: str | int, k: int, metadata: dict=None):
    if query is None and metadata is None:
        raise RuntimeError(f"Retrieval received neither query nor metadata to perform any search")
    ## Handle dataset path
    embeddings = pl.Path(CYTOMINE_CONFIG['embeddings'])
        
    ## Handle query file type
    query_path = embeddings/f'{query}_embedding.pth'
    if query_path.is_file(): # skip encoding step
        print("= Treating query file as pre-encoded embedding")
        query_embedding  = WholeSlideEmbedding().load_embedding(path=query_path).squeeze().numpy()
        print("= Embedding loaded successfully!")
    else: raise RuntimeError(f'No embedding known for query image {query}')
    
    ## Handle subset selection by dataset selection and metadata filtration
    if query:      
        # Perform search        
        if metadata:
            print(f'= Filtering subset by metadata')
            if type(metadata) is dict:
                subset = index.filter_metadata(metadata, None)
                best_imgs, best_sims, best_fns = index.search_subset(np.expand_dims(query_embedding, 0), int(k), subset)
            else:
                raise RuntimeError('Metadata filter must be either dict or path to json file')

        else:
            print(f"= Searching entire database of size {index.ntotal}")
            best_imgs, best_sims, best_fns = index.search(np.expand_dims(query_embedding, 0), int(k))

        result = {
            "query": query,
            "metadata-filter": metadata,
            "embedding_database": str(embeddings),
            "similarities": list(zip(best_imgs, best_sims, best_fns))
        }
        return result
