import tensorflow as tf
import numpy as np
import stm
import ltm
import pymysql
import os,sys,string
import globalvar as gl
#................................
#数据库游标定义成全局的
global conn
global cursor


conn = pymysql.connect(host='localhost',user='root',passwd='cbz997341',db='ltm')
cursor = conn.cursor()
gl._init()
gl.set_value('cs',cursor)
#X = input('please input your goal vector:')
#stm = stm.stm()
#top5 = stm.top5(X)
d_o = ltm.data_opra()
d_o.add_data('pain')


cursor.close()
conn.close()