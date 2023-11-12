import os
from logging import getLogger
from fastapi import FastAPI

from config.env_reader import APIConfig

from src.routers import main_router
logger = getLogger(__name__)

app = FastAPI(
    title=APIConfig.title,
    description=APIConfig.description,
    version=APIConfig.version,
)

app.include_router(main_router.router, prefix="", tags=[""])

