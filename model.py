import theano
import numpy as np

class Memory(object):

    """
    author:liuxianggen
    """
    def __init__(self, **kwargs):
        self.vocab_size = kwargs.pop('vocab_size')
        self.n_in = kwargs.pop('nemb')
        self.n_hids = kwargs.pop('nhids')
        self.n_layer = kwargs.pop('n_layer')
        self.n_label = kwargs.pop('label_size')
        self.params=[]

    def apply(self, facts, facts_mask, question, question_mask, y):

        table = lookup_table(self.n_in, self.vocab_size)
        self.params += table.params

        facts_encoder = auto_encoder(facts, facts_mask, self.vocab_size,
                                     self.n_in, self.n_hids, table=table)
        self.eval = facts_encoder.output
        return self.eval


	# def append(self, entry):
	# 	self.entries.append(entry)
    #
	# def read(self, index):
	# 	return self.entries[index]

#
# class LocationNet(object):
# 	"""
# 	author: Suber
# 	"""
# 	def __init__(self, **kwargs):
# 		self.n_in = kwargs.pop('n_in')
# 		self.n_hids = kwargs.pop('n_hids')
# 		self.n_layer = kwargs.pop('n_layer')
#
# 	def prepare_inputs(self):
# 		pass
#
# 	def apply(self, ques, ques_mask, ht, mem, mem_mask):
# 		gru_in, gru_in_mask = prepare_inputs(ques, ques_mask, ht, mem, mem_mask)
# 		#gru_in should be (n_steps, bat_sizes, embedding)
# 		assert gru_in.ndim == 3
# 		assert gru_in_mask.ndim == 3
# 		lencoder = GRU(self.n_in, self.n_hids, with_contex=False)
# 		dense = denselayer(self.n_in, self.n_hids)
#
# 		#apply
# 		gru_out = lencoder.get_allsteps_out(gru_in, gru_in_mask)
# 		select_w = dense.apply(gru_out)
# 		self.lt=T.nnet.softmax(select_w)
# 		return self.lt
#
#
# class Reasoner(object):
# 	"""the whole model"""
#
# 	def __init__():
