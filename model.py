import theano
import numpy as np

from component import *
class Memory(object):
    """
    author:liuxianggen
    """

    def __init__(self,facts, facts_mask, vocab_size, n_in, n_hids, **kwargs):
        self.vocab_size = vocab_size #24
        self.n_in = n_in #30
        self.n_hids = n_hids  #39
        self.params = []
        self.cost = 0

        if 'table' in kwargs:
            table = kwargs['table']
        else:
            table = lookup_table(n_in, vocab_size)

        # self.params += table.params
        facts_encoder = auto_encoder(facts, facts_mask, self.vocab_size,
                                     self.n_in, self.n_hids, table=table)
        self.params += facts_encoder.params
        self.output = facts_encoder.output  # (10,5,39)
        self.cost = facts_encoder.cost #a float
        self.cost_entry = facts_encoder.cost_entry # vector of float (5), the cost for the specific fact choosed
        return self.output  # (10,5,39)

    def read(self, index):
        if(index<len(self.cost_entry)):
            return self.output[:,index,:],self.cost_entry[index]
        else:
            None,None



class Question(object):
    """
    author:liuxianggen
    """
    def __init__(self,question, question_mask,  vocab_size, n_in, n_hids, **kwargs):
        self.vocab_size = vocab_size #24
        self.n_in = n_in #30
        self.n_hids = n_hids  #39
        self.params = []

        if 'table' in kwargs:
            table = kwargs['table']
        else:
            table = lookup_table(n_in, vocab_size)

        # self.params += table.params
        facts_encoder = auto_encoder(question, question_mask, self.vocab_size,
                                     self.n_in, self.n_hids, table=table)
        self.params += facts_encoder.params
        self.output = facts_encoder.output  # (13,39)
        self.cost = facts_encoder.cost
        return self.output # (13,39)


# take neural reasoner for example
class Reasoner_lxg(object):

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

        memory = Memory(facts, facts_mask, self.vocab_size,
                                     self.n_in, self.n_hids, table=table)
        quest = Question(question, question_mask, self.vocab_size,
                                         self.n_in, self.n_hids, table=table)


        self.params += memory.params
        self.params += quest.params

        mem = memory.output #(10,5,39)
        que = quest.output #(13,39)

        for _ in range(self.n_layer):
            que = self.interact(mem, que)

        logistic_layer = LogisticRegression(questions_rep,
                                            self.n_hids, self.n_label)
        self.params += logistic_layer.params
        self.cost = logistic_layer.cost(y)/y.shape[0]
        self.decoder_cost = facts_encoder.cost + questions_encoder.cost

        self.errors = logistic_layer.errors(y)
        return self.cost, self.decoder_cost

    def interact(self, facts_rep, questions_rep):
        self.W_f = param_init().orth((self.n_hids, self.n_hids))
        self.W_q = param_init().orth((self.n_hids, self.n_hids)) #(39, 39)
        self.b_f = param_init().constant((self.n_hids,))
        self.b_q = param_init().constant((self.n_hids,))
        self.params += [self.W_f, self.W_q, self.b_f, self.b_q]

        questions_rep = T.tanh(theano.dot(questions_rep, self.W_q) + self.b_q)
        facts_rep = T.tanh(theano.dot(facts_rep, self.W_f) + self.b_f)

        def _one_step(question_rep, facts_rep):
            if question_rep.ndim == 1:
                question_rep = T.shape_padleft(question_rep, n_ones=1)
            inter_rep = (question_rep + facts_rep).max(axis=0)
            return inter_rep

        inter_reps, updates = theano.scan(_one_step,
                                          sequences=questions_rep,
                                          outputs_info=None,
                                          non_sequences=facts_rep
                                          )
        return inter_reps

class LocationNet_lxg(object):

	def __init__(self, context, **kwargs):
		self.num_fact =

	# def prepare_inputs():

	def apply(self):
        
		return self.lt