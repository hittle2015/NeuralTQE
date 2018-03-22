from __future__ import division
from collections import Counter
from torch import optim
import os
import sys
import torch
from torch.autograd import Variable
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn as nn
from mtdataloader import *
from metrics import *
from mtvocabulary import *
#use_cuda = torch.cuda.is_available()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# create a file handler
handler = logging.FileHandler('./QEtrain4.log')
handler.setLevel(logging.INFO)

# create a logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)

# add the handlers to the logger
logger.addHandler(handler)

#use_cuda = False
use_cuda = torch.cuda.is_available()
def train_minibatch(input, target, score, model, optimizer, criterion):
	batch_size = len(input)
	optimizer.zero_grad()
	
	input = Variable(torch.LongTensor(input))
	target = Variable(torch.LongTensor(target))
	golden = Variable(torch.FloatTensor(score))

	if use_cuda:
		input = input.cuda()
		target = target.cuda()
		golden = golden.cuda()

	preds = model(input, target, batch_size)
	loss = 0.0
	#for prd, gld in zip(preds, golden):

	loss += criterion(preds, golden)
	# for i, (pred, max_score) in enumerate(zip(preds, [35, 25, 25 ,15])):
	# 	loss += criterion(pred, golden[:,i]/ max_score)

	loss.backward()
	optimizer.step()

	return loss.data[0]


def train(model, vocab, args):
	optimizer = optim.Adam( model.parameters(), lr=args.learning_rate, weight_decay =args.weight_decay)
	criterion = nn.MSELoss()
	data_loader = DataLoader(vocab, args.train_file)

	lowest_loss = 100.00
	sys.stdout.write("\nTraining for %d epochs..." % args.epochs)
	model.train()
	test(model, vocab, args)
	for epoch in range(1, args.epochs + 1 ):
		#idx = 0
		for input, target, score in data_loader.get_batches(args.batch_size):
			#idx += 1
			loss = train_minibatch(input, target, score, model, optimizer, criterion)

			## The commented line below is for identify noisy data that cause the program to crash.
			### 
			#if idx == 941 or idx == 942:
			#	print(input, idx)
			#	for lst in input:
			#		print(vocab.id2src(lst))

			#if idx%10==0:
				#print (loss, idx,'\n')
			sys.stdout.write('\rEpoch[{}] - train loss: {:.3f}\r\r'.format(epoch, loss))
			sys.stdout.flush()
		model.eval()
		result = test(model, vocab, args)
		#print(result)
		#sys.stdout.write('Epoch %d finished: '%(epoch+1))
		#type(result)
		#logger.info (result)
		if result < lowest_loss:
			lowest_loss = result
			torch.save(model, args.save_path)
			logger.info('Saved %s as %s' % (model.__class__.__name__, args.save_path))
	# return train_org, train_pred


def test(model, vocab, args):
	data_loader = DataLoader(vocab, args.test_file)
	criterion = nn.MSELoss()
	loss = 0.0
	tot_size = 0
	
	test_orgs = []
	test_preds = []

	for input, target, score in data_loader.get_batches(args.batch_size, shuffle = False):
		batch_size = len(input)
		tot_size += batch_size
		#print (input)
		input = Variable(torch.LongTensor(input))
		target = Variable(torch.LongTensor(target))
		golden = Variable(torch.FloatTensor(score))
		#print('golden: ', golden)
		if use_cuda:
			input = input.cuda()
			target = target.cuda()
			golden = golden.cuda()

		# preds = model(input, target, batch_size)
		preds = model(input,target, batch_size)
		#print(len(pred))
		#print(pred[0])
		# print(len(input))
		#print('Prediction: ', pred)
		# for prd, gld in zip(preds, golden):
		# 	loss += batch_size*criterion (prd, gld)
		loss += batch_size*criterion (preds, golden)
			#test_orgs.append(gld)




		test_orgs.extend(golden.data.cpu().numpy().tolist()) 
		test_preds.extend(preds.data.cpu().numpy().tolist())



		# for i, (pred, max_score) in enumerate(zip(preds, [35, 25, 25, 15])):
		#    # print i, pred
		# 	#print((pred*max_score).data[:])
		# 	loss[i] += batch_size * criterion(pred*max_score, golden[:, i]).data[0]
		# 	#print(pred*max_score) #, golden[:, i]
		# 	test_orgs[i].extend(golden[:,i].data.cpu().numpy().tolist()) 
		# 	##test_preds[i].extend((pred*max_:score).data[:])
		# 	#print (len((pred*max_score).data.cpu().numpy().tolist()))
		# 	test_preds[i].extend((pred*max_score).data.cpu().numpy().tolist())	
	result = loss / tot_size
	sys.stdout.write ('\n--------------- marking line ---------')
	sys.stdout.write('\n%5.3d instances have been test:' %(len(test_preds)))
	#sys.stdout.write('[Test Performance: MSE ]:\n Usefulness/Transfer: %5.3f; Terminology/Style: %5.3f; Idiomatic Writing: %5.3f; Target Mechanics: %5.3f \n' % (result[0], result[1], result[2], result[3]))
	#
	print('[Test Performance: MSE ]:%5.3f' %(result))
	print("correlation on %s"%('HTER'))
	#print(test_orgs)
	#print(test_preds)
	pr, spr, kt = calc_correl(test_orgs, test_preds)

	print("pr %.3f, spr %.3f, kt %.3f"%(pr, spr, kt))
	with open(args.result_file, 'w') as otfile:
		print ('score'+','+'prediction',file=otfile)
		for org, prd in zip (test_orgs, test_preds):
			print(str(org)+','+str(prd)+'\n', file=otfile)

	return result.data[0]
