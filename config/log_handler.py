from logging.handlers import TimedRotatingFileHandler
from logging import Formatter,FileHandler

formatter = Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')

def time_rotating_handler(file_path:str) -> TimedRotatingFileHandler:
    handler = TimedRotatingFileHandler(filename = file_path, when='midnight',interval=1,encoding='utf-8')
    handler.setFormatter(formatter)
    handler.suffix = "%Y%m%d"
    return handler

def file_handler(file_path:str) -> FileHandler:
    handler = FileHandler(filename=file_path)
    handler.setFormatter(formatter)
    return handler