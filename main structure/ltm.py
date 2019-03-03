import tensorflow as tf
import numpy as np
import globalvar  as gl

def ltm_get_x(X):#接受X，返回所有的ltm的运行结果（weight）
	list = []
	pain = PAIN(X)
	list = list.append((pain.w,'pain'))
	#shame = SHAME(X)
	#list = list.append((shame.w,'shame'))
	return list

class PAIN():
	def __init__(self,X):#先训练模型，后运用
		self.X = X
		self.w = self.compute_w()
	def study(self):#接着最近一次的训练结果继续训练
		self.x = tf.placeholder(tf.float32, shape = (None,4))
		self.W1 = tf.Variable(tf.zeros([4,10]))
		self.b1 = tf.Variable(tf.zeros([10]))
		self.W2 = tf.Variable(tf.zeros([10,1]))
		self.b2 = tf.Variable(tf.zeros([1]))
		self.y1 = tf.nn.softmax(tf.matmul(self.x,self.W1) + self.b1)
		self.y2 = tf.nn.softmax(tf.matmul(self.y1,self.W2) + self.b2)
		self.y_ = tf.placeholder(tf.float32, shape = (None,1))

		self.cross_entropy = -tf.reduce_sum(self.y_*tf.log(self.y2))
		self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(self.cross_entropy)

		saver = tf.train.Saver(max_to_keep = 1)		
		init = tf.global_variables_initializer()

		self.sess = tf.Session()
		self.sess.run(init)
		#saver.restore(self.sess,"C:\\Users\\cbz13\\Desktop\\HDL\\code of brain")
		for i in range(1000):
			batch_xs = np.random.rand(2,4)
			batch_ys = np.random.rand(2,1)
			self.sess.run(self.train_step, feed_dict={self.x:batch_xs, self.y_:batch_ys})
		print(self.sess.run(self.W1))
		saver.save(self.sess,"C:\\Users\\cbz13\\Desktop\\HDL\\code of brain")	
	def compute_w(self):#模型已经训练完，直接调用执行函数
		self.x = tf.placeholder(tf.float32, shape=(1,4))
		self.w1 = tf.Variable(tf.zeros([4,10]))
		self.b1 = tf.Variable(tf.zeros([10]))
		self.w2 = tf.Variable(tf.zeros([10,1]))
		self.b2 = tf.Variable(tf.zeros([1]))
		self.y1 = tf.nn.softmax(tf.matmul(self.x,self.w1) + self.b1)
		self.y2 = tf.nn.softmax(tf.matmul(self.y1,self.w2) + self.b2)
		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.saver = tf.train.Saver(max_to_keep = 1)
		self.sess.run(init)
		self.saver.restore(self.sess, "C:\\Users\\cbz13\\Desktop\\HDL\\code of brain")
		print(self.sess.run(self.y2, feed_dict = {self.x: [self.X]}))
		return self.sess.run(self.y2, feed_dict = {self.x: [self.X]})

class SHAME():

class data_opra():
	def __init__(self):
		self.cursor = gl.get_value('cs')
	def add_data(self,str):#添加数据
			sql = "create table if not exists %s (name varchar(128) primary key, age int(4))"%(str)
			self.cursor.execute(sql)
			data = input('please input name and age:').strip().split(' ')
			sql = "insert into %s(name, age) values ('%s', %d)"%(str,data[0],int(data[1]))
			self.cursor.execute(sql)
			sql = "select * from %s"%(str)
			self.cursor.execute(sql)
			self.cursor.connection.commit()
	def search_data(self,list):
		data = []
		for i in list:
			sql = "select *from" + i[1] + "where "
			self.cursor.execute(sql)
			data = data.append(self.cursor.fetchall())
		return data
