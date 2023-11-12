import logging
from typing import Any
from src.constant import Constants

from src.redis_job.job_manager import JobManager

logger = logging.getLogger(__name__)

class_job_manager = JobManager(Constants.class_type)

caption_job_manager = JobManager(Constants.caption_type)

def make_job_key(photo_id:int) -> str:
    return f"{photo_id}_job"

def add_class_job(photo_id:int,url:str) -> Any:
    job_key = make_job_key(photo_id)
    logger.info(f"register job: {job_key} in {__name__}")
    class_job_manager.add_job(job_key,url)
    logger.info(f"completed register job: {job_key} in {__name__}")
    
    
def add_caption_job(photo_id:int,url:str) -> Any:
    job_key = make_job_key(photo_id)
    logger.info(f"register job: {job_key} in {__name__}")
    caption_job_manager.add_job(job_key,url)
    logger.info(f"completed register job: {job_key} in {__name__}")
