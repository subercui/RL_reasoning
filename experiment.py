import os
import time
import logging
import numpy as np
import theano
from theano import tensor as T
from agent import Agent

config = getattr(configurations, 'get_config')()
# collect n_itr dataset epochs
n_itr = config['n_itr']
# collect n_eps episodes
n_eps = config['n_eps']

agent = Agent(**config)

for _ in xrange(n_itr):
	
	paths = []

	for _ in xrange(N):
		observations = []
		actions = []
		rewards = []


		#this episode finishes, compute all cost, train backward
		agent.f_train(facts[0].T, facts[1].T, 
			question[0].T, question[1].T, label, returns)