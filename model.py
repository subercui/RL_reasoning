import theano
import theano.tensor as T
import numpy as np
from utils import repeat_x

class Memory(object):

	def __init__(self, entries=None):
		"""entries: list"""
		self.entries = []
		assert isinstance(entries, list)
		if entries != None:
			self.entries = entries

	def append(entry):
		self.entries.append(entry)

	def read(index):
		return self.entries[index]


class LocationNet(object):
	"""
	author: Cui Haotian
	"""

	def __init__(self, **kwargs):

		self.n_in = kwargs.pop('n_in')
		self.n_hids = kwargs.pop('n_hids')
		self.n_layers = kwargs.pop('n_layers')


	def prepare_inputs(ques, ht, mem):
		"""
		ques:(1,qsize)
		ht:(1,htsize)
		mem.output:(mem size, embedding size)

		output:(mem size,1,qsize+htsize+embedding )
		"""

        ques = repeat_x(ques, mem.length)
        ht = repeat_x(ht, mem.length)
		output = T.concatenate(ques, ht, mem.output)
		return output

	def apply(self, ques, ht, mem):

		self.length = mem.length
		gru_in = prepare_inputs(ques, ht, mem)
		#gru_in should be (n_steps, bat_sizes, embedding)
		assert gru_in.ndim == 3 # shape=(tstep/mem size/batch size, 1, vector size)
		assert gru_in_mask.ndim == 3

		lencoder = GRU(self.n_in, self.n_hids, with_contex=False)
		gru_out = lencoder.apply(gru_in, mask_below=None) # gru_out shape=(tstep/mem size/batch size, 1, n_hids)
		dense1 = DenseLayer(gru_out.flatten(2), self.n_hids, 1)		
		select_w = dense1.output # shape=(mem size, 1)
		self.lt=T.nnet.softmax(select_w.T)# (1, mem size)
		return self.lt


class Reasoner(object):
	"""the whole model"""

	def __init__():