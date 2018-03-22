import sys
from collections import Counter
import logging
import codecs
import numpy as np
import itertools

class Vocab(object):
	def __init__(self, train_file, pretrained_src = None, pretrained_tgt = None):
		src_cnt = Counter()
		tgt_cnt = Counter()
		with codecs.open(train_file,'r', 'utf-8') as f:
			for line in f.readlines():
				info  = line.strip().split('\t')
				assert len(info) == 3, (line,info )
				src_cnt.update(info[0].split())
				tgt_cnt.update(info[1].split())
		self._id2src = ['UNK', '</s>']
		self._id2tgt = ['UNK', '</s>']
		for w in src_cnt:
			if src_cnt[w]>=2:
				self._id2src.append(w)
		

		for w in tgt_cnt:
			if tgt_cnt[w]>=2:
				self._id2tgt.append(w)
		self.source_vocab_size  =  len(self._id2src)
		self.target_vocab_size  =  len(self._id2tgt)
		#print self.source_vocab_size, self.target_vocab_size

		self._src2id = dict(zip( self._id2src, range(self.source_vocab_size)))
		self._tgt2id = dict(zip( self._id2tgt, range(self.target_vocab_size)))
		self.pretrained_src = pretrained_src
		self.pretrained_tgt = pretrained_tgt

	def src2id(self, x):
		if type(x) is list:
			return [self._src2id.get( t, 0) for t in x]
		return self._src2id.get(x, 0)

	def id2src(self, x):
		if type(x) is list:
			return [self._id2src[t] for t in x]
		return self._id2src[x]

	def tgt2id(self, x):
		if type(x) is list:
			return [self._tgt2id.get( t, 0) for t in x]
		return self._tgt2id.get(x, 0)

	def id2tgt(self, x):
		if type(x) is list:
			return [self._id2tgt[t] for t in x]
		return self._id2tgt[x]

	def get_pretrained_src(self):
		if self.pretrained_src is None:
			return None
		embs = [[]]*len(self._src2id)
		with codecs.open(self.pretrained_src,'r','utf-8') as f:
			for line in f.readlines():
				info = line.strip().split()
				if len(info) !=301: # the length of the line should be equal to the dimension size of the embeddings +1
					continue
				try:
					word, data = info[0], info[1:]
				except IndexError as e:
					continue
				if word in self._src2id:
					embs[self.src2id(word)] = data

		emb_size = len(data)
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		#return embs
		return np.asarray(embs, dtype=np.float32)

	def get_pretrained_tgt(self):
		if self.pretrained_tgt is None:
			return None
		embs = [[]]*len(self._tgt2id)
		with codecs.open(self.pretrained_tgt,'r','utf-8') as f:
			for line in f.readlines():
				info = line.strip().split()
				word, data = info[0], info[1:]
				assert len(data) == 300, (line, len(data))
				if word in self._tgt2id:
					embs[self.tgt2id(word)] = data

		emb_size = len(data)
		for idx, emb in enumerate(embs):
			if not emb:
				embs[idx] = np.zeros(emb_size)
		#return embs
		return np.asarray(embs,dtype = np.float32)
