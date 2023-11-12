
from src.inference.inference_client import InferenceClient
from src.photo.redis_task import class_job_manager
from src.photo.mysql_task import mysql_class_job_manager

def main():
    class_infer = InferenceClient(class_job_manager,mysql_class_job_manager,"cat,prod,dev")
    
    class_infer.prediction_loop()


if __name__ == '__main__':
    main()