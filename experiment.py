# -*- coding:utf-8 -*-
import os
import time
import logging
import numpy as np
import theano
from theano import tensor as T
from dataProcess.stream import preprocess
from agent import Agent
import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# theano.config.exception_verbosity = 'high'
theano.config.optimizer = 'fast_compile'

# config = getattr(config, 'get_config')()
# data_class = preprocess(*config['train_file'])
# for facts, question, label in data_class.data_stream():
# 	print label

if __name__ == '__main__':

    config = getattr(config, 'get_config')()
    # collect n_itr dataset epochs
    n_itr = config['n_itr']
    # collect n_eps episodes
    n_eps = config['n_eps']

    agent = Agent(**config)
    data_class = preprocess(*config['train_file'])
    for _ in xrange(n_itr):

        paths = []

        for facts, question, label in data_class.data_stream():
            observations = []
            actions = []
            rewards = []
            # there should fullfill an episode, takes in facts, questions , labels
            # return answer results, and rewards. The episode including interaction
            # with env is all done in  f_train, which will intricically call reasoner.apply
            rl_cost, sl_cost, decoder_cost = agent.f_train(facts[0].T, facts[1].T, question[0].T, question[1].T, label)

            print 'the costs are: ',rl_cost,sl_cost,decoder_cost

        # facts[0].T.shape, facts[1].T.shape, question[0].T.shape, question[1].T.shape, label.shape layout: (10, 5) (10, 5) (13, 1) (13, 1) (1,)

        # 实际上这是个不好的写法，我们应该尽量减少theano内部内容？比如与环境交互的部分移出去？

        # this episode finishes, compute all cost, train backward
        # for facts, question, label in data_class.data_stream():
        #	sums = agent.f_train(facts[0].T, facts[1].T, question[0].T, question[1].T, label)[1]

# TODO: 先把前传调通，一个一个看变量，再调反传
