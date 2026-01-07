### Ecosystem Imports ###
import argparse
import json
import os
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
### Internal Imports ###
from src.cli_methods.indexing import calculate_embedding_for_image
from api.jobs import enqueue_job
import asyncio
########################
router = APIRouter()

@router.post("/rebuild")
async def rebuild(
    request: Request,
) -> JSONResponse:
    """Rebuild the index from the embedding and meta files in the CYTOMINE_CONFIG['embeddings'] directory.

    Args:
        request (Request): The FastAPI request

    Returns:
        JSONResponse: Status
    """
    app = request.app
    #return JSONResponse(status_code=200, content=calculate_embedding_for_image(request.app.state.index, path, filename, image_id))
    async def rebuild():
        return await asyncio.to_thread(request.app.state.index.rebuild)
    job_id = enqueue_job(app, kind="rebuild", coro_factory=rebuild)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})

@router.post("/restore")
async def restore(
    request: Request,
) -> JSONResponse:
    """Restore the index to its last save

    Args:
        request (Request): The FastAPI request

    Returns:
        JSONResponse: Status
    """
    app = request.app
    #return JSONResponse(status_code=200, content=calculate_embedding_for_image(request.app.state.index, path, filename, image_id))
    async def restore():
        return await asyncio.to_thread(request.app.state.index.restore)
    job_id = enqueue_job(app, kind="restore", coro_factory=restore)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})