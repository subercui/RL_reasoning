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

	def __init__(**kwargs):
		discount = kwargs['discount']
		final_award = kwargs['final_award']
		stp_penalty = kwargs['stp_penalty']

		env = Env(discount, final_award, stp_penalty)
		self.executor = Reasoner(env, **kwargs)

		self.f_train = train()


	def train(self):
		x = T.lmatrix()
	    x_mask = T.matrix()
	    y = T.lmatrix()
	    y_mask = T.matrix()
	    l = T.lvector()
	    # returns_var = T.matrix() # if needed compute this variable outside of f_train

	    self.actions, returns_var, sl_cost, decoder_cost = self.executor.apply(x, x_mask, y, y_mask, l)

		rl_cost = -T.mean(log_likelihood_sym * returns_var)
		cost = combine_costs(sl_cost, decoder_cost, rl_cost)
		grads = theano.grad(cost, agent.params)
		updates = adam(grads, params, learning_rate=learning_rate)

		f_train = theano.function(
			inputs=[x, x_mask, y, y_mask, l, return_var],
			outputs=None,
			updates=updates,
			allow_input_downcast=True
			)

		return f_train


