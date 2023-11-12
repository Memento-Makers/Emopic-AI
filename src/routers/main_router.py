"""
main routing 
"""
from src.routers.request_dto import InferenceRequest
from typing import  Dict
from logging import getLogger
from fastapi import APIRouter, BackgroundTasks, Response

from src.photo import redis_task

router = APIRouter()
logger = getLogger(__name__)

@router.get("/health")
def health() -> Dict[str, str]:
        return {"health", "ok"}
    
@router.post('/classification')
def predict(request_body: InferenceRequest, background_tasks: BackgroundTasks) -> Response:
    
    background_tasks.add_task(redis_task.add_class_job,request_body.photo_id,request_body.url)
    
    return Response(status_code=204)

@router.post('/captioning')
def predict(request_body: InferenceRequest, background_tasks: BackgroundTasks) -> Response:
    
    background_tasks.add_task(redis_task.add_caption_job,request_body.photo_id,request_body.url)
    
    return Response(status_code=204)
