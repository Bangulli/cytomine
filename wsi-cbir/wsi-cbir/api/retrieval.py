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

@router.get("/retrieval")
async def retrieval(
    request: Request,
    query: str,
    staining: str = None,
    organ: str = None,
    species: str = None,
    diagnosis: str = None,
    k: int = 3,
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
    meta = {}
    # NOTE: Index expects AND combinations to be declared by '&' but this character is invalid for URLs so it is replaced by + in cytomine and exchanged here.
    if staining: meta['staining']=staining.replace('AND', '&').replace('OR', '|') ## skip if empty
    if organ: meta['organ']=organ.replace('AND', '&').replace('OR', '|') ## skip if empty
    if species: meta['species']=species.replace('AND', '&').replace('OR', '|') ## skip if empty
    if diagnosis: meta['diagnosis']=diagnosis.replace('AND', '&').replace('OR', '|') ## skip if empty
    return JSONResponse(status_code=200, content=find_k_similar(request.app.state.index, query, k, meta if any(meta) else None))
    # try:
    #     return JSONResponse(status_code=200, content=find_k_similar(query, k, meta if any(meta) else None))
    # except Exception as e:
    #     return JSONResponse(content={'status': f'Failed: {e}'})

    