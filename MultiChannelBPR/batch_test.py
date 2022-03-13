'''
Created on Oct 10, 2018
Tensorflow Implementation of Neural Graph Collaborative Filtering (NGCF) model in:
Wang Xiang et al. Neural Graph Collaborative Filtering. In SIGIR 2019.

@author: Xiang Wang (xiangwang@u.nus.edu)
'''
import metrics as metrics
from load_data import *
import multiprocessing
import heapq
import os
import torch
# import tqdm


torch.backends.cudnn.benchmark = True
cores = multiprocessing.cpu_count() // 2

Ks = args.Ks
data_set = args.dataset

data_generator = Data(f'../Data/{data_set}', batch_size=args.batch_size)
USER_NUM, ITEM_NUM = data_generator.n_users, data_generator.n_items
N_TRAIN, N_TEST = data_generator.n_train, data_generator.n_test

def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = 0.
    return r, auc

def get_auc(item_score, user_pos_test):
    item_score = sorted(item_score.items(), key=lambda kv: kv[1])
    item_score.reverse()
    item_sort = [x[0] for x in item_score]
    posterior = [x[1] for x in item_score]

    r = []
    for i in item_sort:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = metrics.auc(ground_truth=r, prediction=posterior)
    return auc

def ranklist_by_sorted(user_pos_test, test_items, rating, Ks):
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    auc = get_auc(item_score, user_pos_test)
    return r, auc

def get_performance(user_pos_test, r, auc, Ks):
    precision, recall, ndcg, hit_ratio = [], [], [], []

    for K in Ks:
        precision.append(metrics.precision_at_k(r, K))
        recall.append(metrics.recall_at_k(r, K, len(user_pos_test)))
        ndcg.append(metrics.ndcg_at_k(r, K))
        hit_ratio.append(metrics.hit_at_k(r, K))

    return {'recall': np.array(recall), 'precision': np.array(precision),
            'ndcg': np.array(ndcg), 'hit_ratio': np.array(hit_ratio), 'auc': auc}

def test_one_user(x):
    # user u's ratings for user u
    rating = x[0]
    #uid
    u = x[1]
    #user u's items in the training set
    try:
        training_items = data_generator.user_buyitems[u]
    except Exception:
        training_items = []
    #user u's items in the test set
    user_pos_test = data_generator.test_set[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

def test_one_user_train(x):
    # user u's ratings for user u
    rating = x[0]
    # uid
    u = x[1]
    # user u's items in the training set

    training_items = []
    # user u's items in the test set
    user_pos_test = data_generator.user_buyitems[u]

    all_items = set(range(ITEM_NUM))

    test_items = list(all_items - set(training_items))

    r, auc = ranklist_by_heapq(user_pos_test, test_items, rating, Ks)

    return get_performance(user_pos_test, r, auc, Ks)

def evaluate(users_to_test, U, I):
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = args.batch_size
    i_batch_size = args.batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size

        user_batch = test_users[start: end]
        rate_batch = np.matmul(U[user_batch], np.transpose(I))
        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = []
        # for result in tqdm.tqdm(pool.imap_unordered(test_one_user, user_batch_rating_uid), total=u_batch_size):
        #     batch_result.append(result)      
        batch_result = pool.map(test_one_user, user_batch_rating_uid)


        for re in batch_result:
            # result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            # result['hit_ratio'] += re['hit_ratio']/n_test_users
            # result['auc'] += re['auc']/n_test_users

        count += len(batch_result)

    assert count == n_test_users
    pool.close()
    return result

def evaluate_urel(users_to_test, U, I, R):
    # Remember
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = args.batch_size
    i_batch_size = args.batch_size

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        
        user_batch = test_users[start: end]

        rate_batch = np.matmul(U[user_batch] + R[user_batch], np.transpose(I))
        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = []
        # for result in tqdm.tqdm(pool.imap_unordered(test_one_user, user_batch_rating_uid), total=u_batch_size):
        #     batch_result.append(result)      
        batch_result = pool.map(test_one_user, user_batch_rating_uid)


        for re in batch_result:
            # result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            # result['hit_ratio'] += re['hit_ratio']/n_test_users
            # result['auc'] += re['auc']/n_test_users

        count += len(batch_result)

    assert count == n_test_users
    pool.close()
    return result

def evaluate_rel(users_to_test, U, I, u_R, i_R):
    # Remember
    result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
              'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

    pool = multiprocessing.Pool(cores)

    u_batch_size = args.batch_size
    i_batch_size = args.batch_size

    u_R = u_R[0:USER_NUM,:]
    i_R = i_R[0:ITEM_NUM,:]

    test_users = users_to_test
    n_test_users = len(test_users)
    n_user_batchs = n_test_users // u_batch_size + 1

    count = 0

    for u_batch_id in range(n_user_batchs):
        start = u_batch_id * u_batch_size
        end = (u_batch_id + 1) * u_batch_size
        
        user_batch = test_users[start: end]

        rate_batch = np.matmul(U[user_batch] + u_R[user_batch], np.transpose(I + i_R))
        user_batch_rating_uid = zip(rate_batch, user_batch)
        batch_result = []
        # for result in tqdm.tqdm(pool.imap_unordered(test_one_user, user_batch_rating_uid), total=u_batch_size):
        #     batch_result.append(result)      
        batch_result = pool.map(test_one_user, user_batch_rating_uid)


        for re in batch_result:
            # result['precision'] += re['precision']/n_test_users
            result['recall'] += re['recall']/n_test_users
            result['ndcg'] += re['ndcg']/n_test_users
            # result['hit_ratio'] += re['hit_ratio']/n_test_users
            # result['auc'] += re['auc']/n_test_users

        count += len(batch_result)

    assert count == n_test_users
    pool.close()
    return result



