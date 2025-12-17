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
from src.cli_methods.removal import remove_embedding_for_image
########################
router = APIRouter()

@router.post("/rm")
async def removal(
    request: Request,
    image_id: str,
    path: str,
    filename: str,
) -> JSONResponse:
    """

    """
    app = request.app
    #return JSONResponse(status_code=200, content=remove_embedding_for_image(request.app.state.index, path, filename, image_id))
    async def remove():
        return await asyncio.to_thread(remove_embedding_for_image, request.app.state.index, path, filename, image_id)  
    job_id = enqueue_job(app, kind="removal", coro_factory=remove)
    return JSONResponse(status_code=202, content={"job_id": job_id, "state": "queued"})
    
    