'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import random as rd
import scipy.sparse as sp


class Data(object):    
	def __init__(self, path):
		self.path = path

		train_file = path + '/train.txt'
		test_file = path + '/test.txt'

		pv_file = path +'/pv.txt'
		cart_file = path +'/cart.txt'

		# get number of users and items
		self.n_users, self.n_items = 0, 0
		# number of interactions
		self.n_train, self.n_test = 0, 0
		
		# user to item list, item to user list
		self.user_buyitems, self.buyitem_users = {}, {}
		self.user_pvitems, self.pvitem_users = {}, {}
		self.user_cartitems, self.cartitem_users = {}, {}
		self.test_set = {}
		
		self.exist_buyuser, self.exist_pvuser, self.exist_cartuser = [], [], []

		with open(train_file) as f:
			for l in f.readlines():
				if len(l) > 0:
					l = l.strip('\n').split(' ')
					uid = int(l[0])
					items = [int(i) for i in l[1:]]
					self.user_buyitems[uid] = items
					self.exist_buyuser.append(uid)
					for item in items:
						if item in self.buyitem_users:
							self.buyitem_users[item].append(uid)
						else:
							self.buyitem_users[item] = [uid]                           
					self.n_items = max(self.n_items, max(items))
					self.n_users = max(self.n_users, uid)
					self.n_train += len(items)
					#  self.exist_users.append(uid)
					
		with open(pv_file) as f_pv:
			for l in f_pv.readlines():
				if len(l) == 0: break
				l = l.strip('\n').split(' ')
				uid = int(l[0])
				items = [int(i) for i in l[1:]]
				self.user_pvitems[uid] = items
				self.exist_pvuser.append(uid)
				for item in items:
					if item in self.pvitem_users:
						self.pvitem_users[item].append(uid)
					else:
						self.pvitem_users[item] = [uid]                          
				self.n_items = max(self.n_items, max(items))
				self.n_users = max(self.n_users, uid)
				# self.n_train += len(items)

		with open(cart_file) as f_cart:
			for l in f_cart.readlines():
				if len(l) == 0: break
				l = l.strip('\n').split(' ')
				uid = int(l[0])
				items = [int(i) for i in l[1:]]
				self.user_cartitems[uid] = items
				self.exist_cartuser.append(uid)
				for item in items:
					if item in self.cartitem_users:
						self.cartitem_users[item].append(uid)
					else:
						self.cartitem_users[item] = [uid]                           
				self.n_items = max(self.n_items, max(items))
				self.n_users = max(self.n_users, uid)
				# self.n_train += len(items)


		with open(test_file) as f_test:
			for l in f_test.readlines():
				if len(l) == 0: break
				l = l.strip('\n').split(' ')
				try:
					uid = int(l[0])
					items = [int(i) for i in l[1:]]
				except Exception:
					continue           
				self.test_set[uid] = items
				self.n_test += len(items)
					
		# plus 1 , because indexs of items and users start from 0 
		self.n_items += 1
		self.n_users += 1
		
		self.print_statistics()
		
		# self.exist_user = [self.exist_buyuser, self.exist_pvuser, self.exist_cartuser]
		self.user_allact = [self.user_buyitems, self.user_pvitems, self.user_cartitems]
		# self.allact_user = [self.buyitem_users, self.pvitem_users, self.cartitem_users]
		self.test_set = dict(sorted(self.test_set.items()))

	def LoadData(self):
		for i in range(len(self.user_allact)):
			d = self.user_allact[i]
			d = dict(sorted(d.items()))
			row_ind = [k for k, v in d.items() for _ in range(len(v))]
			col_ind = [i for ids in d.values() for i in ids]
			mat = sp.csr_matrix(([2**i]*len(row_ind), (row_ind, col_ind)))
			if i==0:
				trnMat = mat
				buyMat = mat
			# if i== ##:
			# 	buyMat = sp.csr_matrix(([i]*len(row_ind), (row_ind, col_ind)))
			else:
				trnMat = trnMat + mat
		# test set
		tstInt = np.reshape(list(self.test_set.keys()),[-1])
		# tstInt = np.reshape(list(self.test_set.keys()), [-1])
		# tstStat = (tstInt!=None)
		# tstUsrs = np.reshape(np.argwhere(tstStat!=False), [-1])
		tstUsrs = tstInt

		return trnMat, tstInt, buyMat, tstUsrs


	def print_statistics(self):
		print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
		print('buy_interactions=%d' % (self.n_train + self.n_test))
		print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

def negSamp(tembuy, curlist):
	temsize = 1000#1000
	negset = [None] * temsize
	cur = 0
	for temcur in curlist:
		if tembuy[temcur] == 0:
			negset[cur] = temcur
			cur += 1
		if cur == temsize:
			break
	negset = np.array(negset[:cur])
	return negset