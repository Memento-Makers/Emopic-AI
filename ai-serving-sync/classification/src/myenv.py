import os
from dotenv import load_dotenv

load_dotenv()


# docker 내부의 /app/src/.env 파일에 있는 환경변수 읽어오는 함수

def get_port():
    return int(os.getenv('port'))

def get_model_path():
    return os.getenv('model_path')

def get_host():
    return os.getenv('host')
