import tensorflow as tf
import numpy as np
import pymysql
import globalvar as gl

'''导入初始数据'''
s = tf.placeholder(tf.float32,[None,7])
a = [[1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0]]

conn = pymysql.connect(host='localhost',user='root',passwd='cbz997341',db='ltm')
cursor = conn.cursor()
sql = "select * from sa_tuple"
cursor.execute(sql)
string = cursor.fetchall()
S = np.empty(shape=[0,7])
for i in range(30):
	S = np.append(S,[[int(string[i][0][0]),int(string[i][0][1]),int(string[i][0][2]),int(string[i][0][3]),int(string[i][0][4]),int(string[i][0][5]),int(string[i][0][6])]],axis = 0)

'''申请神经网络变量'''
w01 = tf.Variable(tf.random_normal([7,2],stddev = 1,mean = 0,seed = 1))
w13 = tf.Variable(tf.random_normal([2,9],stddev = 1,mean = 0,seed = 1))
w23 = tf.Variable(tf.random_normal([9,9],stddev = 1,mean = 0,seed = 1))

'''定义计算过程'''
node1 = tf.matmul(s,w01)
node2 = tf.matmul(node1,w13)
node3 = tf.matmul(a,w23)
Q = node2 + node3

'''定义损失函数和反向传播过程'''
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

'''建立会话'''
with tf.Session() as sess:
	init_op = tf.global_variables_initializer()
	sess.run(init_op)
	for i in range(30):
		result = sess.run(train_step,feed_dict = {s:S[i:i + 1]})
	
	'''参数入库'''
	cursor.execute(sql)
	cursor.connection.commit()
cursor.close()
conn.close()