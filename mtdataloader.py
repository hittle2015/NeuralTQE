import codecs
import numpy as np
import itertools

class DataLoader(object):
	def __init__(self, vocab, fname):
		self.src_data = []
		self.tgt_data = []
		self.scores =[]
		self.vocab = vocab
		with codecs.open(fname,'r','utf-8') as f:
			i =0
			for line in f.readlines():
				i +=1
				try:
					info  = line.strip().split('\t')
					assert len(info) == 3, line
					src = info[0].split()
					tgt = info[1].split()
					scores = float(info[2])
					self.src_data.append(src)
					self.tgt_data.append(tgt)
					self.scores.append(scores)
					
				except ValueError as e:
					 print ("error",e,"on line",i, info[2])
			

	def get_batches(self, batch_size, shuffle = True):
		idx = list(range(len(self.src_data)))
		if shuffle:
			np.random.shuffle(idx)
		cur_size = 0
		input, target, score = [], [], []
		for _id in sorted(idx, key = lambda x: len(self.src_data[x])):
			cur_size += len(self.src_data[_id])
			input.append(self.src_data[_id])
			target.append(self.tgt_data[_id])
			score.append(self.scores[_id])
			if cur_size  >= batch_size:
				cur_size  = 0
				seq_len = max(len(t) for t in input)
				input = [ self.vocab.src2id(t)+ [0]*(seq_len - len(t)) for t in input ]
				seq_len = max(len(t) for t in target)
				target = [ self.vocab.tgt2id(t) + [0]*(seq_len - len(t)) for t in target ]
				yield input, target, score
				input, target, score = [], [], []
		if len(input)>0:
			seq_len = max(len(t) for t in input)
			input = [ self.vocab.src2id(t)+ [0]*(seq_len - len(t)) for t in input ]
			seq_len = max(len(t) for t in target)
			target = [ self.vocab.tgt2id(t) + [0]*(seq_len - len(t)) for t in target ]
			yield input, target, score
		#print (input)
 
