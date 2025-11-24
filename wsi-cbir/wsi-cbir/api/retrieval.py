### Ecosystem Imports ###
import os
import json
### External Imports ###
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
import argparse
### Internal Imports ###
from src.cli_methods.retrieval import find_k_similar
########################
router = APIRouter()

@router.post("/retrieval")
async def retrieval(
    request: Request,
    query: str,
    datasets: str = None,
    staining: str = None,
    organ: str = None,
    species: str = None,
    diagnosis: str = None,
    k_best: int = 3,
) -> JSONResponse:
    """Find k most similar embeddings for a given image or embedding from a directory

    Args:
        request (Request): The HTTP request
        query (str): Path to the image or embedding used as a query
        embeddings (str): Path to a directory used as a database, must contain indexed.xml. Defaults to 'Embeddings'
        k_best (int, optional): How many images to retrieve. Defaults to 3.

    Returns:
        JSONResponse: A JSON file containing the query image path and the path to the k best matches
    """
    args = argparse.Namespace()
    args.__setattr__('query', query)
    args.__setattr__('embeddings', 'CHIEF')
    args.__setattr__('datasets', datasets)
    meta = {}
    if staining: meta['staining']=staining ## skip if empty
    if organ: meta['organ']=organ ## skip if empty
    if species: meta['species']=species ## skip if empty
    if diagnosis: meta['diagnosis']=diagnosis ## skip if empty
    args.__setattr__('metadata', meta if any(meta) else None)
    args.__setattr__('k_best', k_best)
    args.__setattr__('cpu', True)
    args.__setattr__('return_result', True)
    try:
        result = find_k_similar(args)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        return JSONResponse(status_code=500, content={'error': str(e)})
    