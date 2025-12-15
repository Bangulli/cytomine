### Ecosystem Imports ###
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
### External Imports ###
from fastapi import FastAPI
### Internal Imports ###
from api import indexing, retrieval, removal
from src.retrieval.index import Index
from src.config import CYTOMINE_CONFIG
from src.networks.encoder_mgmt import DIMS
import pathlib as pl
import sched
import time
########################
@asynccontextmanager
async def lifespan(local_app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan of the app."""
    local_app.state.index = Index(pl.Path(CYTOMINE_CONFIG['embeddings']), DIMS[CYTOMINE_CONFIG['encoder']]) if not (pl.Path(CYTOMINE_CONFIG['embeddings'])/'index.faiss').exists() else Index(pl.Path(CYTOMINE_CONFIG['embeddings'])).load()
    index_scheduler = sched.scheduler(time.time, time.sleep)
    def save():
        index_scheduler.enter(CYTOMINE_CONFIG['index_saving_interval'], 1, save)
        local_app.state.index.save()
    index_scheduler.enter(CYTOMINE_CONFIG['index_saving_interval'], 1, save)
    index_scheduler.run(False)
    yield
    map(index_scheduler.cancel, index_scheduler.queue)
    local_app.state.index.save()

PREFIX = "/api"

app = FastAPI(
    title="HES-SO Slide Level Content Based Image Retrieval Server",
    description="HES-SO Slide Level Content Based Image Retrieval Server (CBIR) HTTP API.",
    lifespan=lifespan,
    license_info={
        "name": "Apache 2.0",
        "identifier": "Apache-2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },

)
app.include_router(router=indexing.router, prefix=PREFIX)
app.include_router(router=retrieval.router, prefix=PREFIX)
app.include_router(router=removal.router, prefix=PREFIX)