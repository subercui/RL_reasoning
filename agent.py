import os
import time
import logging
import numpy as np
import theano
from theano import tensor as T
from model import Reasoner, Env


def log_likelihood_sym():
	pass


class Agent(object):

	def __init__(**config):
		env = Env()
		self.executor = Reasoner(env, **config)


	def train(self):
		x = T.lmatrix()
	    x_mask = T.matrix()
	    y = T.lmatrix()
	    y_mask = T.matrix()
	    l = T.lvector()

	    steps,self.actions = self.executor.apply()

		rl_cost = -T.mean(log_likelihood_sym * returns_var)
		cost = combine_costs(...)
		grads = theano.grad(cost, agent.params)
		updates = adam(grads, params, learning_rate=learning_rate)

		self.f_train = theano.function(
			inputs=[x, x_mask, y, y_mask, l, return_var],
			outputs=None,
			updates=updates,
			allow_input_downcast=True
			)


