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
from src.cli_methods.project import delete
########################
router = APIRouter()

@router.post("/project/delete")
async def endpoint_delete(
    request: Request,
    project_id: str,
) -> JSONResponse:
    """

    """
    app = request.app
    #return JSONResponse(status_code=200, content=remove_embedding_for_image(request.app.state.index, path, filename, image_id))
    async def deleteProjectIndex():
        return await asyncio.to_thread(delete, project_id)  
    job_id = enqueue_job(app, kind="projectIndex-delete", coro_factory=deleteProjectIndex)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})
    
    