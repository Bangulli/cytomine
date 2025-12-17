### Ecosystem Imports ###
import os
import json
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
from collections.abc import AsyncGenerator, Callable, Awaitable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import asyncio
import pathlib as pl
from fastapi import FastAPI, Request
### Internal Imports ###
from src.cli_methods.retrieval import find_k_similar
########################

class JobState(str, Enum):
    queued = "QUEUED"
    running = "RUNNING"
    done = "DONE"
    failed = "FAILED"

@dataclass
class Job:
    id: str
    kind: str
    state: JobState = JobState.queued
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None


@dataclass
class QueueItem:
    job_id: str
    coro_factory: Callable[[], Awaitable[Any]]  # builds the coroutine when executed

def enqueue_job(app: FastAPI, kind: str, coro_factory) -> str:
    job_id = uuid4().hex
    app.state.jobs[job_id] = Job(id=job_id, kind=kind)
    app.state.queue.put_nowait(QueueItem(job_id=job_id, coro_factory=coro_factory))
    return job_id


router = APIRouter()

@router.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request):
    jobs = request.app.state.jobs
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job id")

    body = {
        "job_id": job.id,
        "kind": job.kind,
        "state": job.state,
        "created_at": job.created_at.isoformat(),
        "started_at": job.started_at.isoformat() if job.started_at else None,
        "finished_at": job.finished_at.isoformat() if job.finished_at else None,
        "error": job.error,
    }

    if job.state in ("QUEUED", "RUNNING"):
        return JSONResponse(status_code=202, content=body)

    if job.state == "FAILED":
        return JSONResponse(status_code=500, content=body)

    return JSONResponse(status_code=200, content=job.result)