
import logging
from src.database.query_client import QueryClient
from src.photo.mysql_task import mysql_class_job_manager
from src.database.execute_query import use_class_result
from config.env_reader import LogConfig
from config.log_handler import time_rotating_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(time_rotating_handler(LogConfig.mysql_class_client))

def main():
    caption_infer = QueryClient(mysql_class_job_manager,use_class_result)
    caption_infer._loop()


if __name__ == '__main__':
    print('start caption query Client')
    main()