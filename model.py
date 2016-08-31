import theano
import theano.tensor as T
import numpy as np
from layers import *
from utils import *

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
        self.length = self.output.shape[0]

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
        self.output = facts_encoder.output  # (39)
        self.cost = facts_encoder.cost


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

class LocationNet_lxg(object):

    def __init__(self, memory, question,n_hids,**kwargs): #(5,39),(39)
        self.W_f = param_init().orth((n_hids, n_hids))
        self.W_classify = param_init().orth((n_hids, 1))  # (39,1)
        self.b_f = param_init().constant((n_hids,))
        self.b_classify = param_init().constant((1,))
        quest_rep = question.output #(1*39)
        quest_rep = T.addbroadcast(quest_rep,0)
        mom_rep = T.tanh(theano.dot(memory.output, self.W_f) + self.b_f)#(5*39)

        # repeat_x(quest_rep,5)
        quest_mom = quest_rep * mom_rep #(5*39)
        mom_rep = T.tanh(theano.dot(quest_mom, self.W_classify) + self.b_classify)  #(5,1)

        self.softmax = T.nnet.softmax(mom_rep) #(5)
    # def prepare_inputs():

    def apply(self):
        return T.argmax(self.softmax, axis=0) # (1,)


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

        mem = memory.output #(5,39)
        que = quest.output #(39)

        for _ in range(self.n_layer):
            que = self.interact(mem, que)

        logistic_layer = LogisticRegression(que,
                                            self.n_hids, self.n_label)
        self.params += logistic_layer.params
        self.cost = logistic_layer.cost(y)/y.shape[0]
        self.decoder_cost = mem.cost + que.cost

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

class Reasoner(object):
    """the whole model"""

    def __init__():
        return
