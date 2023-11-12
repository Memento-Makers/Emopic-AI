import time
import grpc
import logging
from concurrent import futures

from config import model_pb2
from config import model_pb2_grpc
from config.env_reader import ModelConfig
from prediction import predict

from config.env_reader import LogConfig
from config.log_handler import time_rotating_handler

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(time_rotating_handler(LogConfig.class_server))

class Inference(model_pb2_grpc.InferenceServicer):
    def getResult(self, request, context):
        pred = predict(request.url)
        logger.info(f'request come and pred : {pred}')
        return model_pb2.InferenceResponse(result=pred)
       # return super().getResult(request, context)
    

def serving():
    serving_address = f"{ModelConfig.caption_model_host}:{ModelConfig.caption_model_port}"
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=5))
    model_pb2_grpc.add_InferenceServicer_to_server(Inference(),server)
    server.add_insecure_port(serving_address)
    server.start()
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    logger.info("start classification server")
    serving()