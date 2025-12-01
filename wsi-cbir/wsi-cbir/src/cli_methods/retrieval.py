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
def find_k_similar(query: str | int, k: int, metadata: dict=None):
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
        index = Index(embeddings).load()
        
        if metadata:
            print(f'= Filtering subset by metadata')
            if type(metadata) is dict:
                subset = index.filter_metadata(metadata, subset if subset else None)
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

    # searching without image is disabled for now
    # else:
    #     print(f"= Received no query, will fetch only metadata based results")
        
    #     index = Index(embeddings).load()
        
    #     if args.metadata:
    #         print(f'= Filtering subset by metadata')
    #         subset = index.filter_metadata(args.metadata, subset if subset else None)
    #     else: raise RuntimeError('Didnt receive metadata filter or query')
            
    #     result = {
    #         "query": args.query,
    #         "metadata-filter": args.metadata,
    #         "target-dataset": args.datasets,
    #         "embedding_database": args.embeddings,
    #         "filtered": subset
    #     }
    #     if args.save is not None:
    #         with open(args.save if args.save.endswith('.json') else args.save+'.json', 'w') as file:
    #             print(f"""== Writing results to file {args.save if args.save.endswith('.json') else args.save+'.json'}""")
    #             json.dump(result, file, indent=4)
    #     print("--FINISHED--", json.dumps(result))
    #     if hasattr(args, 'return_result'):
    #         return result
    #     else:
    #         return 0
    
        
