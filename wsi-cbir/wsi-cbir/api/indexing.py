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
    return JSONResponse(status_code=200, content=calculate_embedding_for_image(path, filename, image_id))
    # try:
    #     return JSONResponse(content=calculate_embedding_for_image(path, filename, image_id))
    # except Exception as e:
    #     return JSONResponse(content={'status': f'Failed: {e}'})

    
    