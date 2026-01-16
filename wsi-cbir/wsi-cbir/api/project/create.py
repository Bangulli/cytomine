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
import asyncio
from api.jobs import enqueue_job
### Internal Imports ###
from src.cli_methods.project import create
########################
router = APIRouter()

@router.post("/project/create")
async def endpoint_create(
    request: Request,
    project_id: str,
) -> JSONResponse:
    """

    """
    app = request.app
    #return JSONResponse(status_code=200, content=remove_embedding_for_image(request.app.state.index, path, filename, image_id))
    async def createProjectIndex():
        return await asyncio.to_thread(create, project_id)  
    job_id = enqueue_job(app, kind="projectIndex-create", coro_factory=createProjectIndex)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})
    
    