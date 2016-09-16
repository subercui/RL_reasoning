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

if __name__ == '__main__':

	config = getattr(config, 'get_config')()
	# collect n_itr dataset epochs
	n_itr = config['n_itr']
	# collect n_eps episodes
	n_eps = config['n_eps']

	agent = Agent(**config)
	data_class = preprocess(*train_files)
	for _ in xrange(n_itr):
		
		paths = []
        #
		# for _ in xrange(N):
		# 	observations = []
		# 	actions = []
		# 	rewards = []

		#this episode finishes, compute all cost, train backward
		for facts, question, label in data_class.data_stream():
			sums = agent.f_train(facts[0].T, facts[1].T, question[0].T, question[1].T, label)[1]