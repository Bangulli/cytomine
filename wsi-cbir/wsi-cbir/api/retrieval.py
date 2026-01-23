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
import asyncio
### Internal Imports ###
from src.cli_methods.retrieval import find_k_similar
from src.retrieval.index import Index
from src.config import CYTOMINE_CONFIG
from pathlib import Path
from api.jobs import enqueue_job
from src.cli_methods.project import load
########################
router = APIRouter()

@router.post("/retrieval", status_code=202)
async def retrieval(
    request: Request,
    query: str,
    staining: str = None,
    organ: str = None,
    species: str = None,
    diagnosis: str = None,
    project_id: str=None,
    k: int = 3,
    # TODO project: str = None, 
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
    app = request.app
    meta = {}
    # NOTE: Index expects AND combinations to be declared by '&' but this character is invalid for URLs so it is replaced by 'AND' in cytomine and exchanged here.
    if staining: meta['staining']=staining.replace('AND', '&').replace('OR', '|').replace('%20', ' ') ## skip if empty
    if organ: meta['organ']=organ.replace('AND', '&').replace('OR', '|').replace('%20', ' ') ## skip if empty
    if species: meta['species']=species.replace('AND', '&').replace('OR', '|').replace('%20', ' ') ## skip if empty
    if diagnosis: meta['diagnosis']=diagnosis.replace('AND', '&').replace('OR', '|').replace('%20', ' ') ## skip if empty
    #return JSONResponse(status_code=200, content=find_k_similar(request.app.state.index, query, k, meta if any(meta) else None))

    async def retrieve():
        return await asyncio.to_thread(
            find_k_similar,
            request.app.state.index if project_id == None or project_id == '' else load(project_id), 
            query, 
            k, 
            meta if any(meta) else None
        )    
    job_id = enqueue_job(app, kind="retrieval", coro_factory=retrieve)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})