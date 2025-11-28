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
from src.cli_methods.removal import remove_embedding_for_image
########################
router = APIRouter()

@router.post("/rm")
async def indexing(
    request: Request,
    image_id: str,
    path: str,
    filename: str,
) -> JSONResponse:
    """

    """
    return JSONResponse(status_code=200, content=remove_embedding_for_image(path, filename, image_id))
    # try:
    #     return JSONResponse(content=remove_embedding_for_image(path, filename, image_id))
    # except Exception as e:
    #     return JSONResponse(content={'status': f'Failed: {e}'})

    
    