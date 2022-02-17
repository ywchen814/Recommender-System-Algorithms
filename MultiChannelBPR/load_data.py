'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import numpy as np
import numpy.random as rd
import scipy.sparse as sp
from par import *

class Data(object):    
    def __init__(self, path, batch_size):
        self.path = path
        self.batch_size = batch_size

        train_file = path + '/train.txt'
        test_file = path + '/test.txt'

        pv_file = path +'/pv.txt'
        cart_file = path +'/cart.txt'

        # get number of users and items
        self.n_users, self.n_items = 0, 0
        # number of buy interactions
        self.n_train, self.n_test = 0, 0
        # number of pv interactions
        self.n_pv = 0
        # number of cart interactions
        self.n_cart = 0
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
                self.n_pv += len(items)


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
                self.n_cart += len(items)
    

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
        
        self.exist_user = [self.exist_buyuser, self.exist_pvuser, self.exist_cartuser]
        self.user_allact = [self.user_buyitems, self.user_pvitems, self.user_cartitems]
        self.allact_user = [self.buyitem_users, self.pvitem_users, self.cartitem_users]
        self.level_ratio = np.array(args.rel_weight) * np.array([self.n_train, self.n_pv, self.n_cart])
        self.level_ratio = self.level_ratio / sum(self.level_ratio)

    def print_statistics(self):
        print('n_users=%d, n_items=%d' % (self.n_users, self.n_items))
        print('buy_interactions=%d' % (self.n_train + self.n_test))
        print('pv_interactions=%d' % (self.n_pv))
        print('cart_interactions=%d' % (self.n_cart))
        print('n_train=%d, n_test=%d, sparsity=%.5f' % (self.n_train, self.n_test, (self.n_train + self.n_test)/(self.n_users * self.n_items)))

    def sample_init_pair(self):
        # sample (u, i, L)
        rel = rd.choice([0, 1, 2], p = self.level_ratio)
        user_items = self.user_allact[rel]
        user = rd.choice(self.exist_user[rel])
        pos_item = rd.choice(user_items[user])
        return user, pos_item, rel

    def sample_unobserved_item(self, user, mode='uniform'):
        if mode == 'uniform':
            obs_item = np.concatenate((self.user_allact[0][user],
            self.user_allact[1][user],self.user_allact[2][user]))
            neg_item = rd.choice(np.setdiff1d(np.arange(self.n_items), obs_item))
        return neg_item

    def sample_neg_item_from_rel(self, user, rel):
        # sample (j, N)
        negrel_idx = np.array(args.rel_order) > rel
        # if the interaction already is the least importance,
        # then sample the unobserved items
        if sum(negrel_idx)==0: 
            neg_item = self.sample_unobserved_item(user)
        else:
            neg_rel = rd.choice([0, 1, 2][negrel_idx], p = self.level_ratio[negrel_idx])
            user_items = self.user_allact[neg_rel]
            neg_item = rd.choice(user_items[user])
        return neg_item

    def random_sample(self, neg_num = 1):
        users, pos_items, neg_items = [], [], []
        for i in range(int(np.round(self.batch_size/neg_num))):
            user, pos_item, rel = self.sample_init_pair() 
            for idx in range(neg_num):
                u = rd.uniform(0,1)        
                if args.beta > u:
                    neg_item = self.sample_unobserved_item(user)
                else:
                    neg_item = self.sample_neg_item_from_rel(user, rel)
                users.append(user)
                pos_items.append(pos_item)
                neg_items.append(neg_item)     
        return users, pos_items, neg_items