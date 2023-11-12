import logging
from src.constant import Constants

from src.redis_job.job_manager import JobManager

logger = logging.getLogger(__name__)

mysql_class_job_manager = JobManager(Constants.class_result_type)

mysql_caption_job_manager = JobManager(Constants.caption_result_type)

