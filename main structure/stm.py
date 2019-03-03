import sys
import heapq
import ltm
#利用heapq实现返回weight最大的五个
class TopKHeap():
	def __init__(self):
		self.data = []
	def push(self, elem):
		heapq.heappush(self.data, elem)
	def topk(self):
		list = heapq.nlargest(5,self.data)
		self.data = list
		return list

#内置函数choose，接受一个参数list，list是一个二元组，每次接受一个ltm的标签和权重
#choose函数返回的是当前排序的前5位
#实例化时候要设置为全局的，以便实时更新排序信息
class stm:
	def __init__(self):
		self.heap = heap_sort.TopKHeap()
	def choose(X):
		list = ltm.ltm_get_x(X)
		for tuple in list:
			self.heap.push(tuple)
		return self.heap.topk()