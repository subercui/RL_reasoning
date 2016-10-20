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


if __name__ == '__main__':

    config = getattr(config, 'get_config')()
    # collect n_itr dataset epochs
    n_itr = config['n_itr']
    # collect n_eps episodes
    n_eps = config['n_eps']
    # Time out
    Time = config['T']

    agent = Agent(**config)
    data_class = preprocess(*config['train_file'])
    for _ in xrange(n_itr):

        cnt = 0

        for facts, question, label in data_class.data_stream():
            observations = [facts, question]
            actions = []
            rewards = []

            for t in xrange(Time):
                action，terminal = agent.policy.get_action(Mem，Que)
                reward = env.step(action)
                # observations.append(observation)
                actions.append(action)
                rewards.append(reward)
                # observation = next_observation
                if terminal:
                    # Finish rollout if terminal state reached
                    break

                    # We need to compute the empirical return for each time step along the
                    # trajectory
            returns = []
            return_so_far = 0
            for t in range(len(rewards) - 1, -1, -1):
                return_so_far = rewards[t] + discount * return_so_far
                returns.append(return_so_far)
            # The returns are stored backwards in time, so we need to revert it
            returns = returns[::-1]

            paths.append(dict(
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                returns=np.array(returns)
            ))

        observations = np.concatenate([p["observations"] for p in paths])
        actions = np.concatenate([p["actions"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])

        f_train(observations, actions, returns)
        print('Average Return:', np.mean([sum(p["rewards"]) for p in paths]))

        # this episode finishes, compute all cost, train backward
        # for facts, question, label in data_class.data_stream():
        #	sums = agent.f_train(facts[0].T, facts[1].T, question[0].T, question[1].T, label)[1]

# TODO: 先把前传调通，一个一个看变量，再调反传
