import sys
from stream import preprocess
import theano
import theano.tensor as T
from reasoning import Reason
import logging
from utils import adadelta
import configurations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



if __name__ == '__main__':
    #########prepare######
    config = getattr(configurations, 'get_config')()
    train_files = config['train_file']
    test_files = config['test_file']


    #####################
    x = T.lmatrix()
    x_mask = T.matrix()
    y = T.lmatrix()
    y_mask = T.matrix()
    l = T.lvector()

    rate = float(sys.argv[1])

    reason = Reason(**config)
    cost, decoder_cost = reason.apply(x, x_mask, y, y_mask, l)
    gcost = cost*(1-rate) + decoder_cost*rate
    errors = reason.errors

    params = reason.params
    for param in params:
        gcost += T.sum(param ** 2) * 1e-6
        gcost += T.sum(abs(param)) * 1e-6
    grads = T.grad(gcost, params)

    updates = adadelta(params, grads)

    fn = theano.function([x, x_mask, y, y_mask, l], [errors], updates=updates)
    # fn = theano.function([x, x_mask, y, y_mask, l], shapes, updates=updates)
    test_fn = theano.function([x, x_mask, y, y_mask, l], [errors])

    for _ in range(300):
        data_class = preprocess(*train_files)
        test_class = preprocess(*test_files)
        i=0
        sums= 0
        for facts, question, label in data_class.data_stream():
            sums += fn(facts[0].T, facts[1].T, question[0].T, question[1].T, label)[0]
            i+=label.shape[0]
        logger.info("train err {}".format(float(sums)/float(i)))
        i=0
        sums= 0
        for facts, question, label in test_class.data_stream():
            sums += test_fn(facts[0].T, facts[1].T, question[0].T, question[1].T, label)[0]
            i+=label.shape[0]
        logger.info("test err {}".format(float(sums)/float(i)))


