
from src.database.query_client import QueryClient
from src.photo.mysql_task import mysql_caption_job_manager
from src.database.execute_query import use_caption_result

def main():
    caption_infer = QueryClient(mysql_caption_job_manager,use_caption_result)
    caption_infer._loop()


if __name__ == '__main__':
    main()