"""
redis client configuration
"""

import redis
from config.env_reader import RedisConfig

my_redis_client = redis.Redis(
    host=RedisConfig.redis_host,
    port=RedisConfig.redis_port,
    db=RedisConfig.redis_db,
    decode_responses=RedisConfig.redis_decode_responses
)