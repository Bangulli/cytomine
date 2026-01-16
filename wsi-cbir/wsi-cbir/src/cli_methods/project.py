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
import logging
log = logging.getLogger("uvicorn.error")

########################

def create(project_id: str):
    """Create a subindex for a project

    Args:
        project_id (str): The project ID
    """
    log.info(f"Creating project index for {project_id}")
    os.mkdir(pl.Path(CYTOMINE_CONFIG['embeddings'])/project_id)
    index = Index(pl.Path(CYTOMINE_CONFIG['embeddings'])/project_id, DIMS[CYTOMINE_CONFIG['encoder']])
    index.save()

def add(project_id:str, image_id:str):
    """Add an image to a project subindex

    Args:
        project_id (str): The project ID
        image_id (str): The base image ID
    """
    log.info(f"Adding {image_id} to {project_id}")
    index = Index(pl.Path(CYTOMINE_CONFIG['embeddings'])/project_id, DIMS[CYTOMINE_CONFIG['encoder']]).load()
    with open(pl.Path(CYTOMINE_CONFIG['embeddings'])/f"{image_id}_meta.json", "r") as file:
        meta = json.load(file)
        emb = tc.load(pl.Path(CYTOMINE_CONFIG['embeddings'])/f"{image_id}_embedding.pth", weights_only=False)
        _ = index.add(emb.unsqueeze(0), [image_id], meta['meta'], meta['filename'])
    index.save()

def rm(project_id:str, image_id:str):
    """Remove an image from a project subindex

    Args:
        project_id (str): The project ID
        image_id (str): The base image ID
    """
    log.info(f"Removing {image_id} from {project_id}")
    index = Index(pl.Path(CYTOMINE_CONFIG['embeddings'])/project_id, DIMS[CYTOMINE_CONFIG['encoder']]).load()
    index.rm(image_id)
    index.save()

def delete(project_id:str):
    """Delete a project subindex

    Args:
        project_id (str): The project ID
    """
    log.info(f"Deleting project index for {project_id}")
    os.rmdir(pl.Path(CYTOMINE_CONFIG['embeddings'])/project_id)

def load(project_id:str):
    """Loads a project subindex

    Args:
        project_id (str): The project ID

    Returns:
        Index: The project subindex
    """
    log.info(f"Loading project index for {project_id}")
    return Index(pl.Path(CYTOMINE_CONFIG['embeddings'])/project_id, DIMS[CYTOMINE_CONFIG['encoder']]).load()
