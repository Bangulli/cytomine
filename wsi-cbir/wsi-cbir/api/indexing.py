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
#from src.cli_methods.indexing import calculate_embeddings
########################
router = APIRouter()

@router.post("/indexing")
async def indexing(
    request: Request,
    name: str,
    embeddings: str,
    level: int = 1,
    full_precision: bool = False,
    remove_bg: str = 'dilated-otsu',
) -> JSONResponse:
    """Batch processing function for indexing a directory of images

    Args:
        request (Request): The incoming HTTP request
        name (str): Name of to the input dataset directory.
        embeddings (str): Path to the embeddings directory.
        datareader (str, optional): Which datareader to use. Defaults to 'openslide'.
        patch_size (int, optional): Patch size in pixels. Defaults to 512.
        patch_stride (int, optional): Patch stride in pixels. Defaults to 512.
        encoder (str, optional): Which encoder to use. Defaults to 'ProvGigaPath'.
        level (int, optional): Which resolution level to use. Defaults to 1.
        full_precision (bool, optional): If true runs on fp32, significantly slower. Defaults to False.
        remove_bg (bool, optional): If true uses otsu`s method to remove the background. Defaults to False.

    Returns:
        JSONResponse: A JSON file containing a success message
    """
    raise NotImplementedError()

    
    