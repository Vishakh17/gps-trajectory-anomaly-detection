"""Health check endpoint."""

from fastapi import APIRouter

from app.config import get_settings
from app.schemas.trajectory_schema import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", app_name=settings.app_name)
