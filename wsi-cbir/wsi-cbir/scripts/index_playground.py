## Playground script to test functions of src.retrieval.index.Index outside docker.

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import faiss
# from src.networks.encoder_mgmt import DIMS
# from src.utils.metadata_filtration import load_filter_deps, get_meta_with_codes
import numpy as np
import json
import pathlib as pl
import random
import time, torch
from src.config import CYTOMINE_CONFIG
from src.networks.encoder_mgmt import DIMS
from src.retrieval.index import Index
import time

if __name__ == '__main__':
    os.makedirs(pl.Path('./data/wsi-cbir/embeddings'), exist_ok=True)
    index = Index(pl.Path('./data/wsi-cbir/embeddings'), DIMS[CYTOMINE_CONFIG['encoder']]) if not (pl.Path('./data/wsi-cbir/embeddings/index.faiss')).exists() else Index(pl.Path('./data/wsi-cbir/embeddings')).load()
    #print('Index health check init', index.is_healthy, index.ntotal)
    query = torch.load("/home/lorenz/Repositories/cytomine/data/wsi-cbir/embeddings/IMAGE_0_embedding.pth", weights_only=False).unsqueeze(0)
    k = 10
    metafilter = {
        "staining": "antibody & 12710003"
    }
    subset = index.filter_metadata(metafilter, None)
    start = time.time()
    print(index.search_subindex(query, k, subset))
    print(f"Subindex search for {index.ntotal} database took {time.time()-start}s when {len(subset)} items fulfill the condition")
    start = time.time()
    print(index.search_subset(query, k, subset))
    print(f"Subset search for {index.ntotal} database took {time.time()-start}s when {len(subset)} items fulfill the condition")

    """Building subset according to {'staining': 'antibody & 12710003'}
== 999363 samples fulfill the filter conditions {'staining': 'antibody & 12710003'}
(<map object at 0x7d4785707670>, [0.0, 116.94465637207031, 118.9051513671875, 119.16189575195312, 119.74383544921875, 120.5997543334961, 121.42955017089844, 121.64583587646484, 121.88225555419922, 121.93451690673828], <map object at 0x7d47857077c0>)
Subindex search for 3000000 database took 451.3773500919342s when 999363 items fulfill the condition
(<map object at 0x7d4a6c813730>, [0.0, 116.94465637207031, 118.9051513671875, 119.16189575195312, 119.74383544921875, 120.5997543334961, 121.42955017089844, 121.64583587646484, 121.88225555419922, 121.93451690673828], <map object at 0x7d4a6cac61a0>)
Subset search for 3000000 database took 777.4056017398834s when 999363 items fulfill the condition
    """