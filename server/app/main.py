import asyncio
import logging
from contextlib import asynccontextmanager
from enum import Enum

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routers import audio_router, audio_cleanup_coroutine

logger = logging.getLogger("uvicorn.error")

class EventType(Enum):
    DONE = "done"
    STEM = "stem"
    PROGRESS = "progress"
    STATUS = "status"
    MESSAGE = "message"
    UNKNOWN = "unknown"

class HealthCheckResponse(BaseModel):
    status: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    task = asyncio.create_task(
        audio_cleanup_coroutine(
            stem_ttl_seconds=settings.stem_ttl_seconds,
            cleanup_interval=settings.cleanup_interval
        )
    )
    logger.info("Started stem cleanup loop")
    yield

    # Shutdown
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(audio_router)

@app.get("/health_check", response_model=HealthCheckResponse, tags=["health_check"])
async def health_check():
    return HealthCheckResponse(status="ok")

if __name__ == '__main__':
    uvicorn.run(app, log_level="trace")
