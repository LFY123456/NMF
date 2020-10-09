import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from random import choice
import math
# from math import log

class NMF:
	factor = 20
	lr = 0.01
	lamda = 0.01
	max_iter = 200
	threshold = 1e-4
	minVal = 1.0
	maxVal = 5.0
	user = {}
	item = {}
	id2user = {}
	id2item = {}
	u_i_r = defaultdict(dict)
	i_u_r = defaultdict(dict)

	train_data_path = '../data/ML100k/ml100k_train.csv'
	valid_data_path = '../data/ML100k/ml100k_valid.csv'
	test_data_path = '../data/ML100k/ml100k_test.csv'

	def init_model(self):
		self.train_matrix, self.globalmean = self.generate_hash()
		self.P = np.random.random((len(self.user), self.factor))
		self.Q = np.random.random((len(self.item), self.factor))
		self.loss, self.last_valid_loss, self.last_delta_loss = 0.0, 0.0, 0.0

	def generate_hash(self):
		globalmean = 0.0
		rating_num = 0
		for index, line in enumerate(self.trainSet()):
			user_id, item_id, rating = line
			globalmean += rating
			rating_num += 1
			self.u_i_r[user_id][item_id] = rating
			self.i_u_r[item_id][user_id] = rating
			if user_id not in self.user:
				self.user[user_id] = len(self.user)
				self.id2user[self.user[user_id]] = user_id
			if item_id not in self.item:
				self.item[item_id] = len(self.item)
				self.id2item[self.item[item_id]] = item_id
		globalmean = globalmean / rating_num

		train_matrix = np.zeros((len(self.user), len(self.item)))
		for index, line in enumerate(self.trainSet()):
			user_id, item_id, rating = line
			train_matrix[self.user[user_id]][self.item[item_id]] = rating
		return train_matrix, globalmean

	
	def trainSet(self):
		with open(self.train_data_path, 'r') as f:
			for index, line in enumerate(f):
				u, i, r = line.strip('\r\n').split(',')
				r = (float(r) - self.minVal) / (self.maxVal - self.minVal) + 0.01
				yield (int(u), int(i), float(r))

	def valid_test_Set(self, path):
		with open(path, 'r') as f:
			for index, line in enumerate(f):
				u, i, r = line.strip('\r\n').split(',')
				r = (float(r) - self.minVal) / (self.maxVal - self.minVal) + 0.01
				yield (int(u), int(i), float(r))

	def predict(self, user_id, item_id):
		if(self.containUser(user_id) and self.containItem(item_id)):
			pu = self.P[self.user[user_id], :]
			qi = self.Q[self.item[item_id], :]
			return np.dot(pu, qi)
		elif(self.containUser(user_id) and not self.containItem(item_id)):
			return sum(self.u_i_r[user_id].values()) / float(len(self.u_i_r[user_id]))
		elif(self.containItem(item_id) and not self.containUser(user_id)):
			return sum(self.i_u_r[item_id].values()) / float(len(self.i_u_r[item_id]))
		else:
			return self.globalmean

	def cal_rmse(self, listdata):
		error = 0.0
		count = 0
		for entry in listdata:
			error += abs(entry[2] - entry[3]) ** 2
			count += 1
		if(count == 0):
			return error
		return math.sqrt(float(error) / count)


	def valid_test_model(self, path):
		valid_test_loss = 0.0
		pre_true = []
		for index, line in enumerate(self.valid_test_Set(path)):
			user_id, item_id, rating = line
			pre = self.predict(user_id, item_id)
			valid_test_loss += 0.5 * (rating - pre) ** 2
			pre_true.append([user_id, item_id, rating, pre])
		rmse = self.cal_rmse(pre_true)
		valid_test_loss += 0.5 * self.lamda * ((self.P * self.P).sum() + (self.Q * self.Q).sum())
		return rmse, valid_test_loss

	def train_model(self):
		iteration = 0
		last_loss = 0.0
		while(iteration < self.max_iter):
			self.loss = 0.0
			iteration += 1
			# for index, line in enumerate(self.trainSet()):
			# 	user_id, item_id, rating = line
			# 	pu = self.P[self.user[user_id]]
			# 	qi = self.Q[self.item[item_id]]
			# 	error = rating - np.dot(pu, qi)
			# 	self.loss += 0.5 * (error ** 2)
			# 	self.P[u] 


			error = self.train_matrix - np.dot(self.P, self.Q.T)
			self.loss = np.sum(error * error)
			a = np.dot(self.P.T, self.train_matrix)
			b = np.dot(self.P.T, np.dot(self.P, self.Q.T))
			self.Q.T[b != 0] = (self.Q.T * a / b)[b != 0]
			c = np.dot(self.train_matrix, self.Q)
			d = np.dot(self.P, np.dot(self.Q.T, self.Q))
			self.P[d != 0] = (self.P * c / d)[d != 0]

			self.loss += 0.5 * self.lamda * ((self.P * self.P).sum() + (self.Q * self.Q).sum())

			
			delta_loss = self.loss - last_loss
			valid_rmse, valid_loss = self.valid_test_model(self.valid_data_path)
			print('iteration %d: loss = %.5f, delta_loss = %.5f, valid_loss = %.5f, valid_rmse = %.5f' %(iteration, self.loss, delta_loss, valid_loss, valid_rmse))

			if(abs(delta_loss) < self.threshold):
				break
			if(self.loss > last_loss and iteration != 1):
				break
			if(valid_loss > self.last_valid_loss and iteration != 1):
				break
			last_loss = self.loss
			self.last_valid_loss = valid_loss
		test_rmse, test_loss = self.valid_test_model(self.test_data_path)
		print('test RMSE = %.5f' %(test_rmse))

	def containUser(self, user_id):
		if user_id in self.user:
			return True
		else:
			return False

	def containItem(self, item_id):
		if item_id in self.item:
			return True
		else:
			return False


	def main(self):
		# 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2,0.4, 0.6, 0.8, 1
		# 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
		for sele_para in [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1]:
			self.lr = sele_para
			print('self.lr = ', sele_para)
			self.init_model()
			self.train_model()



if __name__ == '__main__':
    nmf = NMF()
    nmf.main()