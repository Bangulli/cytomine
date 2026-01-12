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
import time
from src.config import CYTOMINE_CONFIG
from src.networks.encoder_mgmt import DIMS
from src.retrieval.index import Index

if __name__ == '__main__':
    os.makedirs(pl.Path('./data/wsi-cbir/embeddings'), exist_ok=True)
    index = Index(pl.Path('./data/wsi-cbir/embeddings'), DIMS[CYTOMINE_CONFIG['encoder']]) if not (pl.Path('./data/wsi-cbir/embeddings/index.faiss')).exists() else Index(pl.Path('./data/wsi-cbir/embeddings')).load()
    print('Index health check init', index.is_healthy, index.ntotal)
    index = index.rebuild()
    print('Index health check after rebuild', index.is_healthy, index.ntotal)
    # index.restore()
    # print('Index after restore', index.is_healthy, index.ntotal)
    index.save()