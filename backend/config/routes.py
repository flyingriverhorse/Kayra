"""Configuration endpoints.

© 2025 Murat Unsal — Skyulf Project

Exposes non-sensitive runtime settings to the frontend so client-side UI
can mirror server-side limits (upload size, allowed file types) instead
of hardcoding its own copy, and tell it whether the public demo mode is on.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from backend.config import Settings
from backend.dependencies import get_config

router = APIRouter(tags=["config"])


class UploadConfig(BaseModel):
    """Client-facing upload limits, sourced from server settings."""

    max_upload_size_bytes: int
    allowed_extensions: list[str]
    demo_mode: bool


@router.get("/config", response_model=UploadConfig)
def get_upload_config(settings: Settings = Depends(get_config)) -> UploadConfig:
    """Return the server-side upload limits and the demo-mode flag for the frontend."""
    return UploadConfig(
        max_upload_size_bytes=settings.MAX_UPLOAD_SIZE,
        allowed_extensions=list(settings.ALLOWED_EXTENSIONS),
        demo_mode=settings.DEMO_MODE,
    )
