import os
import time
import logging
import numpy as np
import theano
from theano import tensor as T
from model import Reasoner, Env
from utils import *


class Agent(object):

	def __init__(self,**kwargs):
		final_award = kwargs['final_award']
		stp_penalty = kwargs['stp_penalty']
		env = Env(final_award, stp_penalty)
		self.executor = Reasoner(env, **kwargs)

		self.f_train = self.train()
		self.learning_rate = 0.5

	def train(self):
		x = T.lmatrix()
		x_mask = T.matrix()
		y = T.lmatrix()
		y_mask = T.matrix()
		l = T.lvector()
		# returns_var = T.matrix() # if needed compute this variable outside of f_train
		rl_cost, sl_cost, decoder_cost = self.executor.apply(x, x_mask, y, y_mask, l)
		cost = self.combine_costs(sl_cost, decoder_cost, rl_cost)
		grads = theano.grad(cost, self.executor.params)
		updates = adadelta(grads, self.executor.params, learning_rate= self.learning_rate)

		f_train = theano.function(
			inputs=[x, x_mask, y, y_mask, l],
			outputs=cost,
			updates=updates,
			allow_input_downcast=True
			)

		return f_train

	def combine_costs(sl_cost, decoder_cost, rl_cost):
		return sl_cost+decoder_cost+rl_cost


