

from typing import Any
from src.redis_job.redis_queue import RedisQueue

class JobManager(RedisQueue):
    def __init__(self,key_name:str):
        super().__init__(key_name)
        self.type = self.key
        
    def make_result_key(self,job_key:str) -> str:
        return f"{job_key}_{self.type}"
    
    def add_job(self,job_key:str,data:Any)->Any:
        # 이미지가 등록되어 있는지 확인
        if not self.redis.exists(job_key):
            # 등록되어 있지 않다면 job_key에 해당하는 url 등록
            self.redis.set(job_key,data)
         # queue에 job_key 등록
        self.push_left(job_key)
        
        result_key = self.make_result_key(job_key)
        
        self.set_result(result_key)
    
    def re_register_job(self,job_key:str) -> Any:
        
        self.push_left(job_key)
    
    def set_result(self,result_key:str,result:Any = "")-> Any:
        self.redis.set(result_key,result)  
              
    def get_job(self)->Any:
        # 큐가 비어있지 않다면
        if not self.isEmpty():
            # 다음 job_key 반환
            return self.pop_right()
        else:
            return None
    
    def get_data_from_redis(self,job_key:str) -> Any:
        # job_key를 통해 url 획득
        return self.redis.get(job_key)
        

