import argparse
import os
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
	type=float, 
	default=0.001, 
	help="learning rate")
parser.add_argument("--lamda", 
	type=float, 
	default=0.0001, 
	help="model regularization rate")
parser.add_argument("--batch_size", 
	type=int, 
	default=1024, 
	help="batch size for training")
parser.add_argument("--epochs", 
	type=int,
	default=1000,  
	help="training epoches")
parser.add_argument("--Ks", 
	type=int, 
	default = [1,5,10,20,50,80,100],
	help="compute metrics@top_k")
parser.add_argument("--factor_num", 
	type=int,
	default=256, 
	help="predictive factors numbers in the model")
parser.add_argument("--num_ng", 
	type=int,
	default=4, 
	help="sample negative items for training")
parser.add_argument("--test_num_ng", 
	type=int,
	default=99, 
	help="sample part of negative items for testing")
parser.add_argument("--out", 
	default=True,
	help="save model or not")
parser.add_argument("--gpu", 
	type=str,
	default="0",  
	help="gpu card ID")

args = parser.parse_args(args=[])
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True