# -*- coding:utf-8 -*- 
import sys
sys.path.insert(0, "./dataProcess/")
from stream import preprocess

import os
import time
import logging
import numpy as np
import theano
from theano import tensor as T
from agent import Agent
import config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import config

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
			print facts[0].T
			#there should fullfill an episode, takes in facts, questions , labels
			#return answer results, and rewards. The episode including interaction
			# with env is all done in  f_train, which will intricically call reasoner.apply
			agent.f_train(facts[0].T, facts[1].T, question[0].T, question[1].T, label)

			#实际上这是个不好的写法，我们应该尽量减少theano内部内容？比如与环境交互的部分移出去？

		#this episode finishes, compute all cost, train backward
		#for facts, question, label in data_class.data_stream():
		#	sums = agent.f_train(facts[0].T, facts[1].T, question[0].T, question[1].T, label)[1]