from __future__ import division
from collections import Counter
import torch
import torch.nn as n
from mtnetworks import *
#from scipy.stats.import pearsonr, spearmanr, kendalltau
from metrics import *
import torch.nn.functional as F
import sys
import logging
import codecs
import numpy as np
import itertools
from  mtvocabulary import *
from mtdataloader import *
#import model
from mttrain import *

import argparse
import random
seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


use_cuda = torch.cuda.is_available()
#use_cuda = False
def main(args):
	vocab = Vocab(args.train_file, args.src_emb, args.tgt_emb)
	model = AttentionRegression(vocab, args.emb_size, args.feature_size, args.window_size, args.dropout, args.hidden_size, args.n_layers, args.attention_size)
	#use_cuda = args.use_cuda
	#use_cuda = False
	if use_cuda:
		model.cuda()
	# starting training, comment this line if you are load a pretrained model
	if not args.test:
		train(model, vocab, args)
	else:
		model.eval()
		model = torch.load('translation_quality3.pt', map_location=lambda storage, loc: storage)
		model.eval()
		test(model,vocab, args)

if __name__ == "__main__":
	argparser = argparse.ArgumentParser()
	argparser.add_argument('--emb_size', type = int, default= 300)
	argparser.add_argument('--feature_size', type = int, default= 200)
	argparser.add_argument('--window_size', nargs = '+', type = int, default= [2])
	argparser.add_argument('--dropout', type=float, default=0.1)
	argparser.add_argument('--hidden_size', type = int, default=200)
	argparser.add_argument('--n_layers', type = int, default= 1)
	argparser.add_argument('--attention_size', type = int, default= 150)
	argparser.add_argument('--train_file', default= '../quality_estimation/mttrain')
	argparser.add_argument('--test_file', default= '../quality_estimation/mttest')
	argparser.add_argument('--learning_rate', type = float, default= 0.001)
	argparser.add_argument('--epochs', type = int, default= 1024)
	argparser.add_argument('--batch_size', type = int, default= 512)
	argparser.add_argument('--src_emb', type=str, default= './data/wiki.de.vec')
	argparser.add_argument('--tgt_emb', type=str, default= './data/glove.en.vec')
	argparser.add_argument('--result_file', type=str, default= '../quality_estimation/mt_test_predictions_300.csv')
	argparser.add_argument('--save_path', type=str, default= './mt_translation_quality_emb300.pt')
	argparser.add_argument('--test', type=bool, default= False)
	argparser.add_argument('--weight_decay', type=float, default=1e-3, help='setting up a float')
	#argparser.add_argument('--use_cuda', action='store_true', default=False, help='enables CUDA training')
	args, extra_args = argparser.parse_known_args()
	#args.use_cuda = not args.no_cuda and torch.cuda.is_available()
	main(args)

