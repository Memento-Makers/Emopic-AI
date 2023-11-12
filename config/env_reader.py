"""
enviroments
"""
import os
from dotenv import load_dotenv

load_dotenv()

class LogConfig:
    caption_server = os.getenv("CAPTION_SERVER_LOG_PATH","caption/log/caption.log")
    class_server = os.getenv("CLASS_SERVER_LOG_PATH","classification/log/class.log")
    caption_client = os.getenv("CAPTION_CLIENT_LOG_PATH","src/inference/log/caption_client.log")
    class_client = os.getenv("CLASS_CLIENT_LOG_PATH","src/inference/log/class_client.log")
    mysql_class_client = os.getenv("MYSQL_CLASS_CLIENT_LOG_PATH","src/database/log/mysql_class_client.log")
    mysql_caption_client = os.getenv("MYSQL_CAPTION_CLIENT_LOG_PATH","src/database/log/mysql_caption_client.log")

class RedisConfig:
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_db = int(os.getenv("REDIS_DB", 0))
    redis_decode_responses = bool(os.getenv("REDIS_DECODE_RESPONSES", True))

    
class APIConfig:
    title = os.getenv("API_TITLE", "Emopic Inference Server")
    description = os.getenv("API_DESCRIPTION", "emopic ML serving ")
    version = os.getenv("API_VERSION", "0.1")

class ProcessConfig:
    caption_process_num = int(os.getenv("CAPTION_PROCESS_NUM", 2))
    class_process_num = int(os.getenv("CLASS_PROCESS_NUM", 2))
    
class MysqlBatchConfig:
    caption_batch_num = int(os.getenv("CAPTION_BATCH_NUM", 5))
    class_batch_num = int(os.getenv("CLASS_BATCH_NUM", 10))
    
    
class MysqlConfig:
    mysql_host = os.getenv("MYSQL_HOST", "localhost")
    mysql_port = int(os.getenv("MYSQL_PORT", 3306))
    mysql_user = os.getenv("MYSQL_USER", "root")
    mysql_password = os.getenv("MYSQL_PASSWORD", "")
    mysql_db = os.getenv("MYSQL_DB", "emopic")

class ModelConfig:
    class_model_host = os.getenv("CLASS_MODEL_HOST", "localhost")
    class_model_port = int(os.getenv("CLASS_MODEL_PORT", 8001))
    caption_model_host = os.getenv("CAPTION_MODEL_HOST", "localhost")
    caption_model_port = int(os.getenv("CAPTION_MODEL_PORT", 8002))
    class_model_path= os.getenv("CLASS_MODEL_PATH","resources/coco_XL_model.pth")
    caption_model_path= os.getenv("CAPTION_MODEL_PATH","resources/rf_model.pth")
    caption_token_path= os.getenv("CAPTION_TOKEN_PATH","demo_coco_tokens.pickle")
    
class TranslatorConfig:
    deepl_auth_key = os.getenv("DEEPL_AUTH_KEY","your_key")