
from src.inference.inference_client import InferenceClient
from src.photo.redis_task import caption_job_manager
from src.photo.mysql_task import mysql_caption_job_manager

def main():
    caption_infer = InferenceClient(caption_job_manager,mysql_caption_job_manager,"testsetes")
    caption_infer.prediction_loop()


if __name__ == '__main__':
    main()