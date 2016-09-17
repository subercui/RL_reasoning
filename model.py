import theano
import theano.tensor as T
import numpy as np

from utils import *
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
        self.output = facts_encoder.output  # (5,39)
        self.cost = facts_encoder.cost #a float
        self.cost_entry = facts_encoder.cost_entry # vector of float (5), the cost for the specific fact choosed
        self.length = self.output.shape[0]

    def read(self, index):
        if(index<len(self.cost_entry)):
            return self.output[index:index+1,:],self.cost_entry[index]
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

        # self.params += table.params  //add already
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

        #self.n_in = kwargs.pop('n_in')
        self.n_hids = kwargs.pop('n_hids')
        self.n_layers = kwargs.pop('n_layers')


    def _prepare_inputs(ques, ht, mem):
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

    # wrong, the gru needs be build in the initial function----liuxianggen
    def apply(self, ques, ht, mem):

        self.length = mem.length
        gru_in = self._prepare_inputs(ques, ht, mem)
        #gru_in should be (n_steps, bat_sizes, embedding)
        assert gru_in.ndim == 3 # shape=(tstep/mem size/batch size, 1, vector size)

        lencoder = GRU(gru_in.shape[2], self.n_hids, with_contex=False)
        gru_out = lencoder.apply(gru_in, mask_below=None) # gru_out shape=(tstep/mem size/batch size, 1, n_hids)
        dense1 = Dense(gru_out.flatten(2), self.n_hids)

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
        que = quest.output #(1,39)

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
        self.params = [self.dense.params, self.stop_sig.params]


    def step_forward(self, qfvector, state_tm1=None, init_flag=False):
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
    """
    author: Cui Haotian
    the whole model
    """

    def __init__(self, env, **kwargs):
        self.env = env
        self.vocab_size = kwargs.pop('vocab_size')
        self.n_in = kwargs.pop('nemb')
        self.n_qf = 2*self.n_in
        self.n_grus = kwargs.pop('n_grus')
        self.n_hts = kwargs.pop('n_hts')
        self.n_lhids = kwargs.pop('n_lhids')
        self.n_layer = kwargs.pop('n_layer')
        self.n_label = kwargs.pop('label_size')
        self.T = kwargs.pop('T')
        self.stp_thrd = kwargs.pop('stp_thrd')
        self.params=[]

    def apply(self, facts, facts_mask, question, question_mask, y):

        table = lookup_table(self.n_in, self.vocab_size)
        self.params += table.params

        memory = Memory(facts, facts_mask, self.vocab_size,
                                     self.n_in, self.n_grus, table=table)
        quest = Question(question, question_mask, self.vocab_size,
                                         self.n_in, self.n_grus, table=table)

        self.params += memory.params
        self.params += quest.params

        self.exct_net = Reasoner_RNN(self.n_qf, self.n_hts, self.n_label)
        self.params += self.exct_net.params

        self.loc_net = LocationNet(n_hids=self.n_lhids,n_layers=1)
        self.params += self.loc_net.params


        #init operations
        mem = memory.output #Fact Memory (5,39)
        que = quest.output #(1,39)
        l_idx = 0
        state_tm1 = None


        for t in xrange(T):
            sf, _ = memory.read(l_idx) #(1,39)
            qf = T.concatenate([que, sf], axis = 1)
            ht, stop, answer = self.exct_net.step_forward(qf, state_tm1, init_flag=(t==0))
            state_tm1 = ht
            lt = self.loc_net.apply(que, ht, mem)
            l_idx = T.argmax(lt).flatten()
            #state_tm1, stop, answer, l_idx = _step(memory, l_idx, que, state_tm1)
            terminal = self._terminate(stop)
            reward = self.env.step(answer, terminal, y)
            if terminal:
                break

        
        self.decoder_cost = mem.cost + que.cost

        return self.cost, self.decoder_cost

    # def _step(self,memory, l_idx, que, state_tm1):
    #
    #     sf, _ = memory.read(l_idx) #(1,39)
    #     qf = T.concatenate([que, sf], axis = 1)
    #     ht, stop, answer = self.exct_net.step_forward(qf, state_tm1, init_flag=(t==0))
    #     lt = self.loc_net.apply(que, ht, memory.output)
    #     l_idx = T.argmax(lt).flatten()
    #
    #     return ht, stop, answer, l_idx


    def _terminate(self, stop):
        return stop > self.stp_thrd


class Env(object):

    def __init__(self,discount, final_award, stp_penalty):
        self.discount = discount
        self.final_award = final_award
        self.stp_penalty = stp_penalty

    def step(self, answer, terminal, y):

        if terminal:
            reward = self.final_award*(answer == y)
        else:
            reward = self.stp_penalty

        return reward
        

