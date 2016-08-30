import theano
import numpy as np

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

	def __init__(self, **kwargs):
		self.n_in = kwargs.pop('n_in')
		self.n_hids = kwargs.pop('n_hids')
		self.n_layer = kwargs.pop('n_layer')

	def prepare_inputs():

	def apply(self, ques, ques_mask, ht, mem, mem_mask):
		gru_in, gru_in_mask = prepare_inputs(ques, ques_mask, ht, Mem, mem_mask)
		#gru_in should be (n_steps, bat_sizes, embedding)
		assert gru_in.ndim == 3
		assert gru_in_mask.ndim == 3
		lencoder = GRU(self.n_in, self.n_hids, with_contex=False)
		dense = denselayer(self.n_in, self.n_hids)

		#apply
		gru_out = lencoder.get_allsteps_out(gru_in, gru_in_mask)
		select_w = dense.apply(gru_out)
		self.lt=T.nnet.softmax(select_w)
		return self.lt


class Reasoner_RNN(object):
    def __init__(self, n_qfvector, n_state, n_answer_class):
        self.n_qfvector = n_qfvector
        self.n_state = n_state
        self.n_answer_class = n_answer_class
        self.dense = Dense(self.n_qfvector, self.n_state)
        self.stop_sig = Sigmoid(self.n_state)
        self.answer_sm = Softmax(self.n_state, self.n_answer_class)
        self.dense._init_params()
        self.stop_sig._init_params()
        self.answer_sm._init_params()

    def _step_forward(self, qfvector, state_tm1=None, init_flag=False):
        init_state = T.alloc(numpy.float32(0.), self.n_state)
        if init_flag:
            state_tm1 = init_state
        state = self.dense.apply(qfvector, state_tm1)
        self.state = state
        stop = self.stop_sig.apply(self.state)
        self.stop = stop
        answer = self.answer_sm.apply(self.state)
        self.answer = answer
        return self.state, self.stop, self.answer