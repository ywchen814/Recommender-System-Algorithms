from tkinter import N
import torch
import torch.nn as nn
from datetime import datetime
from batch_test import *
import time

class AIR(nn.Module):
	def __init__(self, user_num, item_num, factor_num):
		super(AIR, self).__init__()
		"""
		user_num: number of users;
		item_num: number of items;
		factor_num: number of predictive factors.
		"""		
		self.embed_user = nn.Embedding(user_num, factor_num)
		self.embed_item = nn.Embedding(item_num, factor_num)

		nn.init.normal_(self.embed_user.weight, std=0.01)
		nn.init.normal_(self.embed_item.weight, std=0.01)

	def forward(self, user, item, pos_user, pos_item, neg_user, neg_item):
		user = self.embed_user(user)
		item = self.embed_item(item)

		pos_user = self.embed_user(pos_user)
		pos_item = self.embed_item(pos_item)

		neg_user = self.embed_user(neg_user)
		neg_item = self.embed_item(neg_item)

		g = self.g_func(user, item)
		g_pos = self.g_func(pos_user, pos_item)
		g_neg = self.g_func(neg_user, neg_item)
		x_hat = torch.mul(g, g_pos-g_neg).sum(dim=1)
                
		loss = - sum(torch.log((torch.sigmoid(x_hat))))
		reg_loss = (sum(user.norm(dim=1, p=2)) + sum(item.norm(dim=1, p=2)) +
            sum(pos_user.norm(dim=1, p=2)) + sum(pos_item.norm(dim=1, p=2)) + 
            sum(neg_user.norm(dim=1, p=2)) + sum(neg_item.norm(dim=1, p=2)))
		
		return loss, reg_loss * args.lamda
	
	def g_func(self, a, b):
		return(a + b)

if __name__ == '__main__':

	users_to_test = list(data_generator.test_set.keys())
	model_name = 'AIR'

	#dir
	if not os.path.exists('result'):
		os.mkdir(f'result')
	if not os.path.exists(f'result/{data_set}'):
		os.mkdir(f'result/{data_set}')

	model_index = rd.randint(1,100)
	while os.path.exists(f'result/{data_set}/AIR_{model_index}'):
		model_index = rd.randint(1,100)
	torchsave = f'AIR_{model_index}'

	# output text
	now = datetime.now()
	now_time = now.strftime("%Y/%m/%d %H:%M:%S")
	model_text = '{} ------- {}_{}\nfactor_num={} batch_size={} lr={} lamda={} beh_ratio ={} neg_num={})'.format(
		now_time, model_name, model_index, args.factor_num, 
			args.batch_size, args.lr, args.lamda, args.beh_ratio, args.neg_num)
	epochrec_path = f'./result/{data_set}/epoch_output.txt'
	bestrec_path = f'./result/{data_set}/best_perf.txt'
	best_result_text = ''

	with open(epochrec_path,'a+') as f:
		f.write(f'\n{model_text}\n')


	# create model
	model = AIR(USER_NUM, ITEM_NUM, args.factor_num)
	model.cuda()
	optimizer = torch.optim.Adam(
				model.parameters(), lr=args.lr)
	# training
	batch_num = data_generator.n_train//args.batch_size + 1

	#some initial var
	best_result = 0
	conver_count = 0
	record = True

	for epoch in range(1, args.epochs + 1):
		model.train() 
		start_time = time.time()
		epoch_loss, epoch_reg_loss = 0, 0
		for batch_idx in range(batch_num):
			user, item, pos_user, pos_item, neg_user, neg_item, _, _ = data_generator.random_sample()

			user = torch.tensor(user).cuda()
			item = torch.tensor(item).cuda()
			pos_user = torch.tensor(pos_user).cuda()
			pos_item = torch.tensor(pos_item).cuda()
			neg_user = torch.tensor(neg_user).cuda()
			neg_item = torch.tensor(neg_item).cuda()

			model.zero_grad()
			loss, reg_loss = model(user, item, pos_user, pos_item, neg_user, neg_item)
			Loss = loss + reg_loss
			Loss.backward()
			optimizer.step()

			epoch_loss += loss/batch_num
			epoch_reg_loss += reg_loss/batch_num

		elapsed_time = time.time() - start_time
		epoch_rec = "Epoch {:d} [{:.0f}s] loss: {:.4f} + {:.4f}".format(epoch, elapsed_time, loss, reg_loss)

		if epoch % 10 == 0:
			print('Evaluate....')
			model.eval()
			with torch.no_grad():
				U = model.embed_user.weight.detach().cpu().numpy()
				I = model.embed_item.weight.detach().cpu().numpy()
				ret = evaluate(users_to_test, U, I)
				recall = '\nRecall: {}\n'.format(np.round(ret['recall'],5))
				ndcg = 'NDCG: {}'.format(np.round(ret['ndcg'],5))
				epoch_rec += recall + ndcg

			if ret['recall'][2] > best_result:
				print('Save the best....')
				best_result = ret['recall'][2]
				best_result_text = epoch_rec
				torch.save({
					'epoch': epoch,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict(),
					'loss': loss,
					}, f'result/{data_set}/{torchsave}.pt')
			else:
				conver_count +=1
				if conver_count == 10:
					break
		
		print(epoch_rec)
		
		with open(epochrec_path,'a+') as f:
			f.write(f'{epoch_rec}\n')


	with open(bestrec_path,'a+') as f:
		f.write(f'\n{model_text}\n')
		f.write(f'{best_result_text}\n')
