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

@router.post("/indexing")
async def indexing(
    request: Request,
    image_id: str,
    path: str,
    filename: str,
) -> JSONResponse:
    """
    Docstring for indexing
    
    :param request: The HTTP request
    :type request: Request
    :param image_id: ID of the uploaded file
    :type image_id: str
    :param path: Path to the uploaded file
    :type path: str
    :param filename: Name of the uploaded file
    :type filename: str
    :return: Success message
    :rtype: JSONResponse
    """
    app = request.app
    #return JSONResponse(status_code=200, content=calculate_embedding_for_image(request.app.state.index, path, filename, image_id))
    async def index():
        return await asyncio.to_thread(calculate_embedding_for_image, request.app.state.index, path, filename, image_id)
    job_id = enqueue_job(app, kind="removal", coro_factory=index)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})

    
    