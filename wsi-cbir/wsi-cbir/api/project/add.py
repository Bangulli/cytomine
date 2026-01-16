# add image to project index### Ecosystem Imports ###
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
import asyncio
from api.jobs import enqueue_job
### Internal Imports ###
from src.cli_methods.project import add
########################
router = APIRouter()

@router.post("/project/add")
async def endpoint_add(
    request: Request,
    project_id: str,
    image_id:str,
) -> JSONResponse:
    """Endpoint to add an image to a project subindex

    Args:
        request (Request): The current request
        project_id (str): The project ID
        image_id (str): The base image ID

    Returns:
        JSONResponse: 202 Accepted
    """
    app = request.app
    #return JSONResponse(status_code=200, content=remove_embedding_for_image(request.app.state.index, path, filename, image_id))
    async def addImageToProjectIndex():
        return await asyncio.to_thread(add, project_id, image_id)  
    job_id = enqueue_job(app, kind="projectIndex-add", coro_factory=addImageToProjectIndex)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})
    
    