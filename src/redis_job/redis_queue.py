from src.redis_job.redis_client import my_redis_client
class RedisQueue:
    def __init__(self,key_name:str):
        self.redis = my_redis_client
        self.key = key_name
    # 큐 사이즈 확인
    def size(self) -> int:
        return self.redis.llen(self.key)
    
    def isEmpty(self) -> bool:
        return self.size() == 0
    
    def push_left(self, data):
        self.redis.lpush(self.key, data)
    
    def pop_right(self):
        return self.redis.rpop(self.key)
    
    