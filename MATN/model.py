import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam
from batch_test import * 
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
from datetime import datetime

class Recommender:
	def __init__(self, sess, datas, inpDim):
		self.inpDim = inpDim
		self.sess = sess
		self.trnMat, self.tstInt, self.buyMat, self.tstUsrs = datas
		self.metrics = dict()
		mets = ['Loss', 'preLoss' 'HR', 'NDCG']
		for met in mets:
			self.metrics['Train'+met] = list()
			self.metrics['Test'+met] = list()

	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret

	def run(self):
		self.prepareModel()
		log('Model Prepared')
		best_result = 0
		if args.load_model != None:
			self.loadModel()
			stloc = len(self.metrics['TrainLoss'])
		else:
			stloc = 0
			init = tf.global_variables_initializer()
			self.sess.run(init)
			log('Variables Inited')
		for ep in range(stloc, args.epoch):
			test = (ep % 1 == 0)
			reses = self.trainEpoch()
			log(self.makePrint('Train', ep, reses, test))
			if test:
				ret = self.evaluate()
				epoch_rec = self.test_save(ep, ret)

				if ret['recall'][1] > best_result:
					best_result = ret['recall'][1]
					best_result_text = epoch_rec
				# log(self.makePrint('Test', ep, reses, test))

		self.best_pref_save(best_result_text)

	def best_pref_save(self, best_result_text):
			# output text
		now = datetime.now()
		now_time = now.strftime("%Y/%m/%d %H:%M:%S")
		model_text = '{} ------- \nfactor_num={} batch_size={} lr={} lamda={} neg_num={}'.format(
			now_time, args.latdim, args.batch, args.lr, args.reg, args.negsamp)
		best_perf_path = './result/{}/best_perf.txt'.format(data_set)
		with open(best_perf_path,'a+') as f:
			f.write('{}\n'.format(model_text))
			f.write('{}\n'.format(best_result_text))

		

	def test_save(self, ep, ret):
		epochrec_path = './result/{}/epoch_output.txt'.format(data_set)
		recall = 'Recall: {}\n'.format(np.round(ret['recall'],5))
		ndcg = 'NDCG: {}\n'.format(np.round(ret['ndcg'],5))
		epoch_rec = 'Epoch:{}\n'.format(ep) + recall + ndcg
		with open(epochrec_path,'a+') as f:
			f.write('{}\n'.format(epoch_rec))
		return epoch_rec

	def multiHeadAttention(self, localReps, glbRep, number, numHeads, inpDim):
		query = tf.reshape(tf.tile(tf.reshape(FC(glbRep, inpDim, useBias=True, reg=True), [-1, 1, inpDim]), [1, number, 1]), [-1, numHeads, inpDim//numHeads])
		temLocals = tf.reshape(localReps, [-1, inpDim])
		key = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, numHeads, inpDim//numHeads])
		val = tf.reshape(FC(temLocals, inpDim, useBias=True, reg=True), [-1, number, numHeads, inpDim//numHeads])
		att = tf.nn.softmax(2*tf.reshape(tf.reduce_sum(query * key, axis=-1), [-1, number, numHeads, 1]), axis=1)
		attRep = tf.reshape(tf.reduce_sum(val * att, axis=1), [-1, inpDim])
		return attRep

	def selfAttention(self, localReps, number, inpDim):
		attReps = [None] * number
		stkReps = tf.stack(localReps, axis=1)
		for i in range(number):
			glbRep = localReps[i]
			temAttRep = self.multiHeadAttention(stkReps, glbRep, number=number, numHeads=args.att_head, inpDim=inpDim) + glbRep
			# fc1 = FC(temAttRep, inpDim, reg=True, useBias=True, activation='relu') + temAttRep
			# fc2 = FC(fc1, inpDim, reg=True, useBias=True, activation='relu') + fc1
			attReps[i] = temAttRep#fc2
		return attReps

	def divide(self, interaction):
		ret = [None] * self.intTypes
		for i in range(self.intTypes):
			ret[i] = tf.to_float(tf.bitwise.bitwise_and(interaction, (2**i)) / (2**i))
		return ret

	def mine(self, interaction):
		activation = 'relu'
		V = NNs.defineParam('v', [self.inpDim, args.latdim], reg=True)
		divideLst = self.divide(interaction)
		catlat1 = []
		for dividInp in divideLst:
			catlat1.append(tf.matmul(dividInp,V))
		catlat2 = self.selfAttention(catlat1, number=self.intTypes, inpDim=args.latdim)
		catlat3 = list()
		self.memoAtt = []
		for i in range(self.intTypes):
			resCatlat = catlat2[i] + catlat1[i]
			memoatt = FC(resCatlat, args.memosize, activation='relu', reg=True, useBias=True)
			memoTrans = tf.reshape(FC(memoatt, args.latdim**2, reg=True, name='memoTrans'), [-1, args.latdim, args.latdim])
			self.memoAtt.append(memoatt)

			tem = tf.reshape(resCatlat, [-1, 1, args.latdim])
			transCatlat = tf.reshape(tf.matmul(tem,memoTrans), [-1, args.latdim])
			catlat3.append(transCatlat)

		stkCatlat3 = tf.stack(catlat3, axis=1)

		weights = NNs.defineParam('fuseAttWeight', [1, self.intTypes, 1], reg=True, initializer='zeros')
		sftW = tf.nn.softmax(weights*2, axis=1)
		fusedLat = tf.reduce_sum(sftW * stkCatlat3, axis=1)
		self.memoAtt = tf.stack(self.memoAtt, axis=1)

		lat = fusedLat
		for i in range(2):
			lat = FC(lat, args.latdim, useBias=True, reg=True, activation=activation) + lat
		return lat

	def prepareModel(self):
		self.intTypes = 3
		self.interaction = tf.placeholder(dtype=tf.int32, shape=[None, self.inpDim], name='interaction')
		self.posLabel = tf.placeholder(dtype=tf.int32, shape=[None, None], name='posLabel')
		self.negLabel = tf.placeholder(dtype=tf.int32, shape=[None, None], name='negLabel')
		intEmbed = tf.reshape(self.mine(self.interaction), [-1, 1, args.latdim])
		self.learnedEmbed = tf.reshape(intEmbed, [-1, args.latdim])

		W = NNs.defineParam('W', [self.inpDim, args.latdim], reg=True)
		posEmbeds = tf.transpose(tf.nn.embedding_lookup(W, self.posLabel), [0, 2, 1])
		negEmbeds = tf.transpose(tf.nn.embedding_lookup(W, self.negLabel), [0, 2, 1])
		sampnum = tf.shape(self.posLabel)[1]

		posPred = tf.reshape(tf.matmul(intEmbed, posEmbeds), [-1, sampnum])
		negPred = tf.reshape(tf.matmul(intEmbed, negEmbeds), [-1, sampnum])
		self.posPred = posPred

		self.preLoss = tf.reduce_mean(tf.reduce_sum(tf.maximum(0.0, 1.0 - (posPred - negPred)), axis=-1))
		self.regLoss = args.reg * Regularize(method='L2')
		self.loss = self.preLoss + self.regLoss

		globalStep = tf.Variable(0, trainable=False)
		learningRate = tf.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay, staircase=True)
		self.optimizer = tf.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

	def trainEpoch(self):
		trnMat = self.trnMat
		num = trnMat.shape[0]
		trnSfIds = np.random.permutation(num)[:args.trn_num]
		tstSfIds = self.tstUsrs
		sfIds = np.random.permutation(np.concatenate((trnSfIds, tstSfIds)))
		# sfIds = trnSfIds
		epochLoss, epochPreLoss = [0] * 2
		num = len(sfIds)
		steps = int(np.ceil(num / args.batch))

		for i in range(steps):
			curLst = list(np.random.permutation(self.inpDim))
			st = i * args.batch
			ed = min((i+1) * args.batch, num)
			batchIds = sfIds[st: ed]

			temTrn = trnMat[batchIds].toarray()
			tembuy = self.buyMat[batchIds].toarray()

			temPos = [[None]*(args.posbat*args.negsamp) for i in range(len(batchIds))]
			temNeg = [[None]*(args.posbat*args.negsamp) for i in range(len(batchIds))]
			for ii in range(len(batchIds)):
				row = batchIds[ii]
				posset = np.reshape(np.argwhere(tembuy[ii]!=0), [-1])
				negset = negSamp(tembuy[ii], curLst)
				idx = 0
				if len(posset) == 0:
					posset = np.random.choice(list(range(args.item)), args.posbat)
				for j in np.random.choice(posset, args.posbat):
					for k in np.random.choice(negset, args.negsamp):
						temPos[ii][idx] = j
						temNeg[ii][idx] = k
						idx += 1
			target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
			res = self.sess.run(target, feed_dict={self.interaction: (temTrn).astype('int32'),
				self.posLabel: temPos, self.negLabel: temNeg
				}, options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
			preLoss, regLoss, loss = res[1:]

			epochLoss += loss
			epochPreLoss += preLoss
			# log('Step %d/%d: loss = %.2f, regLoss = %.2f       ' %\
			# 	(i, steps, loss, regLoss), save=False, oneline=True)
		ret = dict()
		ret['Loss'] = epochLoss / steps
		ret['preLoss'] = epochPreLoss / steps
		return ret

	def evaluate(self):
		result = {'precision': np.zeros(len(Ks)), 'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks)),
				'hit_ratio': np.zeros(len(Ks)), 'auc': 0.}

		pool = multiprocessing.Pool(cores)
		count = 0

		trnMat = self.trnMat
		# tstInt = self.tstInt
		ids = self.tstUsrs
		n_test_users = len(ids)
		testbatch = args.batch
		steps = int(np.ceil(n_test_users / testbatch))
		for i in range(steps):
			st = i * testbatch
			ed = min((i+1) * testbatch, n_test_users)
			batchIds = ids[st:ed]
			
			temTrn = trnMat[batchIds].toarray()
			# temTst = tstInt[batchIds]
			# tembuy = self.buyMat[batchIds].toarray()

			preds = self.sess.run(
				self.posPred, 
				feed_dict={self.interaction:temTrn.astype('int32'), 
				self.posLabel: np.repeat([np.arange(args.item)], len(batchIds), axis = 0)}, 
				options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
		
		
			user_batch_rating_uid = zip(preds, batchIds)
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

	def saveHistory(self):
		if args.epoch == 0:
			return
		with open('History/' + args.save_path + '.his', 'wb') as fs:
			pickle.dump(self.metrics, fs)

		saver = tf.train.Saver()
		saver.save(self.sess, 'Models/' + args.save_path)
		log('Model Saved: %s' % args.save_path)

	def loadModel(self):
		saver = tf.train.Saver()
		saver.restore(sess, 'Models/' + args.load_model)
		with open('History/' + args.load_model + '.his', 'rb') as fs:
			self.metrics = pickle.load(fs)
		log('Model Loaded')	

if __name__ == '__main__':
	if not os.path.exists('result'):
		os.mkdir('result')
	if not os.path.exists('result/' + data_set):
		os.mkdir('result/' + data_set)
	
	logger.saveDefault = True
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	log('Start')
	datas = data_generator.LoadData()
	log('Load Data')

	with tf.Session(config=config) as sess:
		recom = Recommender(sess, datas, args.item)
		recom.run()