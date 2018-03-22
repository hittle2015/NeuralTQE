import torch
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn as nn

use_cuda = torch.cuda.is_available()
#use_cuda = False
class EncoderCNNRNN(nn.Module):
	def __init__(self, vocab_size, emb_size, feature_size, window_size, hidden_size, dropout, n_layers=1, pretrained_embs = None):
		super(EncoderCNNRNN, self).__init__()
		self.n_layers = n_layers
		self.hidden_size = hidden_size
		self.dropout = nn.Dropout(dropout)
		self.embedding = nn.Embedding(vocab_size, emb_size)

		#self.embedding.weight.requires_grad = False
		if pretrained_embs is not None:
			self.embedding.weight.data.copy_(torch.from_numpy(pretrained_embs))
			self.conv = nn.ModuleList([ nn.Conv1d(emb_size, feature_size, 2*sz+1, padding = sz) for sz in window_size])
			self.gru = nn.GRU(feature_size*len(window_size), hidden_size, n_layers, batch_first = True)

	def forward(self, input, batch_size):
		embedded = self.embedding(input).permute(0, 2, 1) # batch_size x seq_len x emb_size =>  batch_size x emb_size x seq_len
		feature = [ self.dropout( F.relu(conv(embedded))) for conv in self.conv] # batch_size x emb_size x seq_len => batch_size x feature_size x seq_len

		output = torch.cat(feature, 1).permute(0, 2, 1) # batch_size x feature_size x seq_len => batch_size x seq_len x feature_size
		hidden = self.initHidden(batch_size)
		output, hidden = self.gru(output, hidden)

		return output, torch.mean(output, 1) # batch_size x seq_len x hidden_size,  batch_size x hidden_size

	def initHidden(self, batch_size):	
		result = Variable(torch.zeros(self.n_layers, batch_size, self.hidden_size))
		if use_cuda:
			return result.cuda()
		else:
			return result

class AttentionRegression(nn.Module):
	def __init__(self, vocab, emb_size, feature_size, window_size, dropout, hidden_size, n_layers, attention_size):
		super(AttentionRegression, self).__init__()
		self.encoder_s = EncoderCNNRNN(vocab.source_vocab_size, emb_size, feature_size, window_size, hidden_size, dropout, n_layers, vocab.get_pretrained_src())
		self.encoder_t = EncoderCNNRNN(vocab.target_vocab_size, emb_size, feature_size, window_size, hidden_size, dropout, n_layers, vocab.get_pretrained_tgt())
		self.s2att_s = nn.Linear(hidden_size, attention_size)
		self.t2att_s = nn.Linear(hidden_size, attention_size ,bias = False)
		self.attw_s = nn.Linear(attention_size, 1)
		
		self.t2att_t = nn.Linear(hidden_size, attention_size)
		self.s2att_t = nn.Linear(hidden_size, attention_size ,bias = False)
		self.attw_t = nn.Linear(attention_size, 1)
		self.dropout = nn.Dropout(dropout)
		self.regression = nn.Linear(2*hidden_size, 1)
		#self.regression = nn.ModuleList([nn.Linear(2*hidden_size, 1) for i in range(4)])


	def forward(self, source, target, batch_size):
		output_s, repr_s = self.encoder_s(source, batch_size)
		output_t, repr_t = self.encoder_t(target, batch_size)
		
		weight_s = self.attw_s(F.relu(self.s2att_s(output_s) + self.t2att_s(repr_t).view(batch_size,1,-1)))#batch_size x seq_len
		weight_t = self.attw_t(F.relu(self.t2att_t(output_t) + self.s2att_s(repr_s).view(batch_size,1,-1)))#batch_size x seq_len

		repr_s = torch.sum(weight_s * output_s, 1)
		repr_t = torch.sum(weight_t * output_t, 1)

		repr_st = self.dropout(torch.cat((repr_s, repr_t), 1))
		score = torch.squeeze( self.regression(repr_st))
		#score = [ torch.squeeze( regression(repr_st)) for regression in self.regression]

		return score
