import logging
import grpc
import asyncio
from time import sleep
from concurrent.futures import ProcessPoolExecutor
from src.utils.translator import translate_to_korean
from src.constant import Constants
from src.redis_job.job_manager import JobManager
from config import model_pb2
from config.model_pb2_grpc import InferenceStub
from config.env_reader import ProcessConfig, ModelConfig
from config.env_reader import LogConfig
from config.log_handler import time_rotating_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class InferenceClient:
    def __init__(self,job_manager:JobManager,mysql_manager:JobManager,default_prediction:str):
        logger.info("make inference client")
        self.job_manager = job_manager
        self.mysql_manager = mysql_manager
        self.prediction = default_prediction
        if job_manager.type == Constants.caption_type:
            self.process_num = ProcessConfig.caption_process_num 
            self.model_host = ModelConfig.caption_model_host
            self.model_port = ModelConfig.caption_model_port
            self.translate_bool = True
            logger.addHandler(time_rotating_handler(LogConfig.caption_client))
        else:
            self.process_num = ProcessConfig.class_process_num
            self.model_host = ModelConfig.class_model_host
            self.model_port = ModelConfig.class_model_port
            self.translate_bool = False
            logger.addHandler(time_rotating_handler(LogConfig.class_client))
        
    def _prediction_if_queue_is_not_empty(self,stub:InferenceStub):
        # queue 에는 job_key만 담고 실제 데이터(이미지)는 redis 에 담음
        job_key = self.job_manager.get_job()  # None or job_key
        try:
            if job_key is not None:
                logger.info(f"predict job_key: {job_key}, type: {self.job_manager.type}")
                result_key = self.job_manager.make_result_key(job_key)
                result = self.job_manager.get_data_from_redis(result_key)
                image_url = self.job_manager.get_data_from_redis(job_key)
                if result != "":  # 공백이 아니라면 이미 예측값이 저장되어 있다는 의미
                    return True
                prediction = stub.getResult(model_pb2.InferenceRequest(url=image_url)).result
                if prediction is not None:  # 응답이 성공적으로 오면
                    logger.info(f"{job_key} is success! pred : {prediction} ")
                    if self.translate_bool:
                        prediction = translate_to_korean(prediction)
                    self.job_manager.set_result(result_key, prediction)  # job id에 예측값 등록
                    self.mysql_manager.push_left(result_key)
                else:
                    self.job_manager.re_register_job(job_key)  # 응답이 지연된 경우나 오지 않은 경우 다시 큐에 등록
        except:
            logger.info(f"{job_key} is fail! re_register_job")
            self.job_manager.re_register_job(job_key)  # 응답이 지연된 경우나 오지 않은 경우 다시 큐에 등록
    def _loop(self):
        serving_address = f"{self.model_host}:{self.model_port}"
        channel = grpc.insecure_channel(serving_address)
        stub = InferenceStub(channel)
        while True:
            sleep(1)
            self._prediction_if_queue_is_not_empty(stub)


    # 멀티 프로세스로 기동
    def prediction_loop(self):
        excutor = ProcessPoolExecutor(self.process_num)  # 병렬 연산을 위한 ProcessPoolExecutor
        loop = asyncio.get_event_loop()

        for _ in range(self.process_num):
            asyncio.ensure_future(loop.run_in_executor(excutor, self._loop()))

        loop.run_forever()