from collections.abc import AsyncGenerator, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import asyncio
import pathlib as pl
from fastapi import FastAPI, Request

from api import indexing, retrieval, removal, jobs
from src.retrieval.index import Index
from src.config import CYTOMINE_CONFIG
from src.networks.encoder_mgmt import DIMS


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # --- load index ---
    emb_path = pl.Path(CYTOMINE_CONFIG["embeddings"])
    dims = DIMS[CYTOMINE_CONFIG["encoder"]]
    app.state.index = (
        Index(emb_path, dims)
        if not (emb_path / "index.faiss").exists()
        else Index(emb_path).load()
    )

    # --- shared state for jobs ---
    app.state.jobs: dict[str, jobs.Job] = {}
    app.state.queue: asyncio.Queue[Optional[jobs.QueueItem]] = asyncio.Queue(maxsize=CYTOMINE_CONFIG.get("queue_maxsize", 0) or 0)

    # This lock ensures: only one “index-touching” thing at a time (jobs + periodic save)
    app.state.index_lock = asyncio.Lock()

    stop_event = asyncio.Event()
    app.state.stop_event = stop_event

    async def worker_loop():
        while True:
            item = await app.state.queue.get()
            if item is None:
                app.state.queue.task_done()
                break

            job = app.state.jobs[item.job_id]
            job.state = jobs.JobState.running
            job.started_at = datetime.now(timezone.utc)

            try:
                # If job work is CPU/IO heavy, do it in a thread to keep the event loop responsive
                # Also serialize access to the index
                async with app.state.index_lock:
                    job.result = await item.coro_factory()
                job.state = jobs.JobState.done
            except Exception as e:
                job.state = jobs.JobState.failed
                job.error = repr(e)
            finally:
                job.finished_at = datetime.now(timezone.utc)
                app.state.queue.task_done()

    async def periodic_save():
        interval = CYTOMINE_CONFIG["index_saving_interval"]
        while not stop_event.is_set():
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
            except asyncio.TimeoutError:
                async with app.state.index_lock:
                    await asyncio.to_thread(app.state.index.save)

    worker_task = asyncio.create_task(worker_loop(), name="job-worker")
    save_task = asyncio.create_task(periodic_save(), name="periodic-save")

    yield

    # --- shutdown ---
    stop_event.set()

    # stop worker via sentinel
    try:
        await app.state.queue.put(None)
    except Exception:
        pass

    for t in (save_task, worker_task):
        t.cancel()
    for t in (save_task, worker_task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    # final save
    try:
        async with app.state.index_lock:
            await asyncio.to_thread(app.state.index.save)
    except Exception:
        pass



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
app.include_router(router=jobs.router, prefix=PREFIX)