import pymysql
from datetime import datetime

import pymysql
from datetime import datetime

class mysql():
    def __init__(self) -> None:
        self.time_now = datetime.now()
        time = self.time_now.strftime("%Y_%m_%d")
        self.database_name_1 = 'six_parameter'
        self.database_name_2 = "electrical_measuring_instrument"
        self.database_name_3 = "activate_signal"
        self.tabel_name = 'six_parameter_table'
        self.tabel_name_2 = "electrical_measuring_instrument_table"
        self.tabel_name_3 = "activate_signal_table"
        

    def connect(self):
        # 连接 MySQL 数据库
        self.conn = pymysql.connect(
            host='127.0.0.1',  # 主机名
            port=3306,         # 端口号，MySQL默认为3306
            user='root',       # 用户名
            password='11088jason', # 密码
        )
        # 创建游标对象
        self.cursor = self.conn.cursor()
    
    def connect_database(self):
        # 连接 MySQL 数据库
        self.conn = pymysql.connect(
            host='127.0.0.1',  # 主机名
            port=3306,         # 端口号，MySQL默认为3306
            user='root',       # 用户名
            password='11088jason', # 密码
            database = self.database_name_1,   # 数据库名称
        )
        # 创建游标对象
        self.cursor = self.conn.cursor()
    
    def connect_database_2(self):
        # 连接 MySQL 数据库
        self.conn = pymysql.connect(
            host='127.0.0.1',  # 主机名
            port=3306,         # 端口号，MySQL默认为3306
            user='root',       # 用户名
            password='11088jason', # 密码
            database = self.database_name_2,   # 数据库名称
        )
        # 创建游标对象
        self.cursor = self.conn.cursor()
    
    def connect_database_3(self):
        # 连接 MySQL 数据库
        self.conn = pymysql.connect(
            host='127.0.0.1',  # 主机名
            port=3306,         # 端口号，MySQL默认为3306
            user='root',       # 用户名
            password='11088jason', # 密码
            database = self.database_name_3,   # 数据库名称
        )
        # 创建游标对象
        self.cursor = self.conn.cursor()

    def create_database(self):
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database_name_1}")
        self.cursor.execute(f'use {self.database_name_1}')
    
    def create_database_2(self):
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database_name_2}")
        self.cursor.execute(f'use {self.database_name_2}')
    
    def create_database_3(self):
        self.cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database_name_3}")
        self.cursor.execute(f'use {self.database_name_3}')

    def close_connect(self):
        self.cursor.close()
        self.conn.close()
    
    def creat_tabel(self):
        create_table_query = """
                            CREATE TABLE IF NOT EXISTS %s (
                                date DATE,
                                time TIME,
                                Experiment_Number INT,
                                Parameter_acquisition INT,
                                Electrical_conductivity FLOAT,
                                Residual_chlorine FLOAT,
                                TOC FLOAT,
                                PH FLOAT,
                                Turbidity FLOAT,
                                Particle1 INT,
                                Particle2 INT,
                                Particle3 INT,
                                Particle4 INT
                                
                            ); 
                            """ % self.tabel_name
        self.cursor.execute(create_table_query)
        self.conn.commit()
    
    def creat_tabel_2(self):
        create_table_query = """
                            CREATE TABLE IF NOT EXISTS %s (
                                date DATE,
                                time TIME,
                                Experiment_Number INT,
                                Flowing_current FLOAT
                            ); 
                            """ % self.tabel_name_2
        self.cursor.execute(create_table_query)
        self.conn.commit()
    
    def creat_tabel_3(self): 
        create_table_query = """
                            CREATE TABLE IF NOT EXISTS %s (
                                activate_signal INT
                            ); 
                            """ % self.tabel_name_3
        self.cursor.execute(create_table_query)
        self.conn.commit()
        
if __name__ == "__main__":
    sql = mysql()
    sql.connect()
    sql.create_database_3()
    sql.creat_tabel_3()
    sql.close_connect()