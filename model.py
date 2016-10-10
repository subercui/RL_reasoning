import theano
import theano.tensor as T
import theano.tensor.extra_ops as Ops
from theano.tensor.nnet import categorical_crossentropy
import numpy as np
from layers.layers import DenseLayer,GRU,lookup_table,auto_encoder,LogisticRegression
from utils import *
from component import Dense,Sigmoid,Softmax
TINY = 1e-8

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
        print self.length

    def read(self, index):

        if(T.lt(index,self.cost_entry.shape[0])):
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

        self.n_hids = kwargs.pop('n_hids')
        self.n_layers = kwargs.pop('n_layers')
        self.n_in = kwargs.pop('n_in')

        self.lencoder = GRU(self.n_in, self.n_hids, with_contex=False)
        self.dense1 = DenseLayer(self.n_hids, 1, nonlinearity=None)
        self.params = self.lencoder.params + self.dense1.params

    def _prepare_inputs(self, ques, ht, mem):
        """
        ques:(1,qsize)
        ht:(1,htsize)
        mem.output:(mem size, embedding size)

        output:(mem size,1,qsize+htsize+embedding )
        """
        ques = repeat_x(ques, mem.length)
        ht = repeat_x(ht, mem.length)

        output = T.concatenate([ques, ht, mem.output.dimshuffle((0,'x',1))],axis=2)
        return output

    def apply(self, ques, ht, mem):

        #  self.length = mem.length
        gru_in = self._prepare_inputs(ques, ht, mem)
        #gru_in should be (n_steps, batch_sizes, embedding)
        assert gru_in.ndim == 3 # shape=(tstep/mem size/batch size, 1, vector size)(5,1,10)

        gru_out = self.lencoder.apply(gru_in, mask_below = None) # gru_out shape=(tstep/mem size/batch size, 1, n_hids)
        select_w = self.dense1.get_output_for(gru_out.flatten(2)) # shape=(mem size, 1)
        self.lt = T.nnet.softmax(select_w.T)# (1, mem size)
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
        self.params = self.dense.params + self.stop_sig.params + self.answer_sm.params

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
    #
    # def dist_info_sym(self, obs_var, state_info_vars=None):
    #     return dict(prob=L.get_output(self._l_prob, {self._l_obs: obs_var}))

class Reasoner(object):
    """
    author: Cui Haotian
    the whole model
    """

    def __init__(self, env, **kwargs):
        self.env = env
        self.vocab_size = kwargs.pop('vocab_size')
        self.n_in = kwargs.pop('nemb')
        self.n_grus = kwargs.pop('n_grus')
        self.n_qf = 2 * self.n_grus
        self.n_hts = kwargs.pop('n_hts')
        self.n_lhids = kwargs.pop('n_lhids')
        self.n_layer = kwargs.pop('n_layer')
        self.n_label = kwargs.pop('label_size')
        self.T = kwargs.pop('T')
        self.stp_thrd = kwargs.pop('stp_thrd')
        self.params=[]
        self.paramss = []

    def apply(self, facts, facts_mask, question, question_mask, y):
        """
        layout: (10, 5) (10, 5) (13, 1) (13, 1) (1,)
        return: answer, cost
        """

        table = lookup_table(self.n_in, self.vocab_size)
        self.params += table.params
        self.paramss +=  table.params

        memory = Memory(facts, facts_mask, self.vocab_size,
                                     self.n_in, self.n_grus, table=table)
        quest = Question(question, question_mask, self.vocab_size,
                                         self.n_in, self.n_grus, table=table)

        self.params += memory.params
        self.params += quest.params

        self.exct_net = Reasoner_RNN(self.n_qf, self.n_hts, self.n_label)
        self.params += self.exct_net.params

        self.loc_net = LocationNet(n_hids=self.n_lhids,n_layers=1,n_in=self.n_qf+self.n_hts)
        self.params += self.loc_net.params



        #init operations
        # mem = memory.output #Fact Memory (5,n_grus=4)
        que = quest.output #(1,n_grus=4)

        l_idx = 0
        htm1 = None

        stops_dist = []
        answers_dist = []
        lts_dist = []
        stops = []
        answers = []
        lts = []
        rewards = []

        for t in xrange(self.T):
            sf, _ = memory.read(l_idx) #(1,n_grus=4)
            qf = T.concatenate([que, sf], axis = 1) #layout: (1, 2*n_grus=8)
            ht, stop_dist, answer_dist = self.exct_net.step_forward(qf, htm1, init_flag=(t==0))
            htm1 = ht
            lt_dist = self.loc_net.apply(que, ht, memory)
            l_idx = T.argmax(lt_dist) #hard attention
            #htm1, stop, answer, l_idx = _step(memory, l_idx, que, htm1)


            answer = T.argmax(answer_dist)

            #TODO: implement a real sampling
            terminal = self._terminate(stop_dist[0,0])
            reward = self.env.step(answer, terminal, y)

            stops_dist.append(stop_dist)
            answers_dist.append(answer_dist)
            lts_dist.append(lt_dist)
            stops.append(terminal)
            answers.append(answer)
            lts.append(l_idx)
            rewards.append(reward)

            if T.gt(terminal,0):
                break

        stops_dist = T.concatenate(stops_dist,axis=0)#ndim=2
        answers_dist = T.concatenate(answers_dist,axis=0)#ndim=2
        lts_dist = T.concatenate(lts_dist,axis=0)#ndim=2
        stops = T.stack(stops,axis=0)#ndim=1
        answers = T.stack(answers,axis=0)#ndim=1
        lts = T.stack(lts,axis=0)#ndim=1
        rewards = T.stack(rewards,axis=0)#ndim=1

        self.decoder_cost = memory.cost + quest.cost
        self.sl_cost = T.mean(categorical_crossentropy(answer_dist, y))

        # print(answers_dist.ndim)



        stop_cost=self.log_likelihood_sym(actions_var=stops, dist_info_vars={'prob': stops_dist}) * rewards
        answer_cost=self.log_likelihood_sym(actions_var=answers, dist_info_vars={'prob': answers_dist}) * rewards
        lt_cost=self.log_likelihood_sym(actions_var=lts, dist_info_vars={'prob': lts_dist}) * rewards
        # print(rewards.ndim)
        # print(stop_cost.ndim)
        # print(answer_cost.ndim)

        self.rl_cost = -T.mean(stop_cost+answer_cost+lt_cost)
        #TODO: we need to improve this rl_cost to introduce anti-variance measures

        print self.rl_cost.ndim
        print self.sl_cost.ndim
        print self.decoder_cost.ndim

        return self.rl_cost, self.sl_cost, self.decoder_cost


    def log_likelihood_sym(self, actions_var, dist_info_vars):
        """
        PS: x_var should be the samples from the distributions represented with dist_info_vars
        """
        probs = dist_info_vars["prob"]
        # Assume layout is N * A
        # i=0
        # while(T.lt(i,actions_var.shape[0])):
        #     ress = probs[i,actions_var[i]]
        # res = T.log(ress + TINY)

        oneHot = Ops.to_one_hot(actions_var,probs.shape[1])
        res = T.log(T.sum(probs*T.cast(oneHot,'float32'),axis=-1)+TINY)
        # print probs.ndim
        # print oneHot.ndim
        # print res.ndim
        return res


    def _step(self,memory, l_idx, que, state_tm1,t):
    
         sf, _ = memory.read(l_idx) #(1,39)
         qf = T.concatenate([que, sf], axis = 1)
         ht, stop, answer = self.exct_net.step_forward(qf, state_tm1, init_flag=(t==0))
         lt = self.loc_net.apply(que, ht, memory.output)
         l_idx = T.argmax(lt).flatten()
    
         return ht, stop, answer, l_idx


    def _terminate(self, stop):
        return stop > self.stp_thrd


class Env(object):

    def __init__(self, final_award, stp_penalty):
        self.final_award = final_award
        self.stp_penalty = stp_penalty

    def step(self, answer, terminal, y):

        if T.gt(terminal,0):
            reward = self.final_award*(answer == y)
        else:
            reward = self.stp_penalty

        return reward
        

