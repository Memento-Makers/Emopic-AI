"""
db connection related stuff
"""

from src.database.mysql_client import mysql_connection


def insert_photo_category()->str:
    return "insert IGNORE into photo_category (category_id,photo_id) values((select id from category where name = %s),%s)"
def insert_category()->str:
    return "insert IGNORE into category (name) values(%s)"

def use_class_result(photo_id:int,class_list:str):
    if class_list == "":
        class_list = ["default"]
    else:
        class_list = class_list.split(",")
    conn = mysql_connection.cursor()
    # class_list에 있는 카테고리 항목이 db에 존재하지 않으면 생성
    category_query = insert_category()
    photo_category_query = insert_photo_category()
    photo_category_value = []
    for i in range(len(class_list)):
        photo_category_value.append((class_list[i],photo_id))
        class_list[i] = [class_list[i]]
    # category 생성 query 
    conn.executemany(category_query,class_list)
    # DB에 반영
    mysql_connection.commit()
    conn.executemany(photo_category_query,photo_category_value)
    # DB에 반영
    mysql_connection.commit()
    # 커넥션 종료
    conn.close()
    

def update_photo() -> str:
    # photo_id에 맞는 caption 업데이트
    return "update photo set caption = %s where id = %s"


def update_diary() -> str:
    # photo_id에 맞는 diary 내용 업데이트
    return "update diary set content = %s where photo_id = %s"

def use_caption_result(photo_id:int,caption:str):
    conn = mysql_connection.cursor()
    val = (caption,photo_id)
    photo_query = update_photo()
    diary_query = update_diary()
    conn.execute(photo_query,val)
    conn.execute(diary_query,val)
    # DB에 반영
    mysql_connection.commit()
    # 커넥션 종료
    conn.close()