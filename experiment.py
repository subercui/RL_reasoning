import sys
import config
import theano
import theano.tensor as T
sys.path.insert(0, "./dataProcess/")
from stream import preprocess

import numpy
from utils import *
from model import Memory
from model import Question
from model import LocationNet_lxg
def lxg_test():

    configs = getattr(config, 'get_config')()
    train_files = configs['train_file']
    test_files = configs['test_file']

    n_hids = configs['nhids']   # hidden units
    n_in = configs['nemb']  #
    n_layer = configs['n_layer']
    # the following two is set by the data, when run the stream.py
    vocab_size = configs['vocab_size']


    x = T.lmatrix()
    x_mask = T.matrix()
    y = T.lmatrix()
    y_mask = T.matrix()
    l = T.lvector()

    gcost = 0
    memory = Memory(x, x_mask, vocab_size,n_in, n_hids)
    quest = Question(y, y_mask, vocab_size,n_in, n_hids)
    location = LocationNet_lxg(memory, quest,n_hids)

    gcost = memory.cost

    # add regularization
    params = memory.params




    for param in params:
        gcost += T.sum(param ** 2) * 1e-6  # l2
        gcost += T.sum(abs(param)) * 1e-6  # l1
    grads = T.grad(gcost, params)
    updates = adadelta(params, grads)
    fn = theano.function([x, x_mask, y, y_mask, l], [quest.output, memory.length,location.softmax], updates=updates, on_unused_input='ignore')

    # test_class = preprocess(*test_files)
    for _ in range(100):
        data_class = preprocess(*train_files)
        # sums = 0.0
        i = 0
        for facts, question, label in data_class.data_stream():
            sums = fn(facts[0].T, facts[1].T, question[0].T, question[1].T, label)[0]
            print sums.shape
            break
        #     i += label.shape[0]
        # print "train error: {}".format(float(sums / float(i)))


if __name__=='__main__':
    lxg_test()
