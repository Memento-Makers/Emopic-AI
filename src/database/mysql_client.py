"""
mysql connect configuration
"""
import MySQLdb
from config.env_reader import MysqlConfig

mysql_connection = MySQLdb.connect(
    host=MysqlConfig.mysql_host,
    port=MysqlConfig.mysql_port,
    user=MysqlConfig.mysql_user,
    password=MysqlConfig.mysql_password,
    database=MysqlConfig.mysql_db
)