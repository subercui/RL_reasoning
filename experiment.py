import sys
import theano
import theano.tensor as T
# from utils import adadelta
import config
import numpy
from utils import *
from model import Memory

def lxg_test():

    x = T.lmatrix()
    x_mask = T.matrix()
    y = T.lmatrix()
    y_mask = T.matrix()
    l = T.lvector()
    config = getattr(config, 'get_config')()
    memory = Memory(**config)
    context = memory.apply(x, x_mask, y, y_mask, l)
    print memory.n_in


if __name__=='__main__':
    lxg_test()
