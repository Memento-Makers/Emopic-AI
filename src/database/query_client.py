import logging
from typing import Callable
from time import sleep
from src.redis_job.job_manager import JobManager
from src.constant import Constants
from config.env_reader import LogConfig
from config.log_handler import time_rotating_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class QueryClient:
    def __init__(self,mysql_manager:JobManager,todo:Callable):
        self.mysql_manager = mysql_manager
        self.todo = todo
        if self.mysql_manager.type == Constants.caption_result_type:
            logger.addHandler(time_rotating_handler(LogConfig.mysql_caption_client))
        else:
            logger.addHandler(time_rotating_handler(LogConfig.mysql_class_client))
    def _execute_query_if_queue_is_not_empty(self):
        result_key = self.mysql_manager.get_job() # None or result_key
        if result_key is not None:
            logger.info(f"query result_key: {result_key}, type: {self.mysql_manager.type}")
            result = self.mysql_manager.get_data_from_redis(result_key)
            photo_id = int(result_key.split("_")[0])
            try:
                self.todo(photo_id,result)    
            except:
                logger.error("error execute query")
                # DB에 요청 실패시 큐에 재등록
                self.mysql_manager.push_left(result_key)
            
    def _loop(self):
        while True:
            sleep(1)
            self._execute_query_if_queue_is_not_empty()
        