"""FastAPI application entry point."""

from __future__ import annotations

import logging
import sys

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.routes_data import router as data_router
from app.api.routes_health import router as health_router
from app.api.routes_prediction import router as prediction_router
from app.config import get_settings


def _configure_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )


_configure_logging()
settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "Research prototype for context-aware GPS trajectory anomaly detection. "
        "Isolation Forest outputs indicate potential anomalies only, not confirmed events."
    ),
    version="0.1.0",
)

app.include_router(health_router, prefix="/api")
app.include_router(data_router, prefix="/api")
app.include_router(prediction_router, prefix="/api")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    _request: Request, exc: RequestValidationError
) -> JSONResponse:
    return JSONResponse(status_code=422, content={"detail": exc.errors()})


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(
    _request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    detail = exc.detail
    if not isinstance(detail, str):
        detail = str(detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": detail})


@app.exception_handler(Exception)
async def unhandled_exception_handler(_request: Request, exc: Exception) -> JSONResponse:
    if isinstance(exc, StarletteHTTPException):
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(status_code=exc.status_code, content={"detail": detail})
    logging.getLogger(__name__).exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected server error occurred."},
    )
