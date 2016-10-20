from __future__ import print_function
from functools import reduce
import re
import tarfile

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.utils.data_utils import get_file

from keras.preprocessing.sequence import pad_sequences
import theano
import theano.tensor as T
from layers import nonlinearities
from layers.layers import DenseLayer,GRU,lookup_table
from utils import *

theano.config.optimizer = 'fast_compile'

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    X = []
    Xq = []
    Y = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    return pad_sequences(X, maxlen=story_maxlen), pad_sequences(Xq, maxlen=query_maxlen), np.array(Y)


EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 40
print('Embed / Sent / Query = {}, {}, {}'.format(EMBED_HIDDEN_SIZE, SENT_HIDDEN_SIZE, QUERY_HIDDEN_SIZE))

try:
    path = get_file('babi-tasks-v1-2.tar.gz', origin='https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise
tar = tarfile.open(path)
# Default QA1 with 1000 samples
# challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples
challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# QA2 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'
train = get_stories(tar.extractfile(challenge.format('train')))
test = get_stories(tar.extractfile(challenge.format('test')))

vocab = sorted(reduce(lambda x, y: x | y, (set(story + q + [answer]) for story, q, answer in train + test)))
# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

X, Xq, Y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tX, tXq, tY = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

print('vocab = {}'.format(vocab))
print('X.shape = {}'.format(X.shape))
print('Xq.shape = {}'.format(Xq.shape))
print('Y.shape = {}'.format(Y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')

class sentrnn(object):
    def __init__(self, inputs):
        self.n_in = EMBED_HIDDEN_SIZE
        self.vocab_size = vocab_size
        self.params = []

        table = lookup_table(self.n_in, self.vocab_size)
        self.params += table.params

        self.out = table.apply(inputs)

class qrnn(object):
    def __init__(self,inputs):
        self.n_in = EMBED_HIDDEN_SIZE
        self.vocab_size = vocab_size
        self.n_hids = EMBED_HIDDEN_SIZE
        self.params = []

        table = lookup_table(self.n_in, self.vocab_size)
        self.params += table.params

        q_emb = table.apply(inputs)

        gru = GRU(self.n_in, self.n_hids, with_contex=False)
        self.params += gru.params

        q_gru = gru.apply(q_emb, mask_below=None)[-1]

        assert q_gru.ndim == 2
        q_gru = q_gru.dimshuffle(('x',0, 1))
        self.out = T.extra_ops.repeat(q_gru, story_maxlen, axis=0)



class Model(object):
    def __init__(self, sent, q):
        self.sentrnn = sentrnn(sent)
        self.qrnn = qrnn(q)
        self.n_in = EMBED_HIDDEN_SIZE
        self.n_hids = EMBED_HIDDEN_SIZE
        self.vocab_size = vocab_size
        self.params = self.sentrnn.params + self.qrnn.params

        sq = self.sentrnn.out + self.qrnn.out

        gru = GRU(self.n_in, self.n_hids, with_contex=False)
        self.params += gru.params

        sq_gru = gru.apply(sq, mask_below=None)[-1]

        dense = DenseLayer(self.n_hids, self.vocab_size, nonlinearity=nonlinearities.softmax)
        self.params += dense.params

        self.out = dense.get_output_for(sq_gru)

sent = T.lmatrix('X')
ques = T.lmatrix('Xq')
label = T.matrix('Y')

model = Model(sent,ques)

cost = T.mean(T.nnet.categorical_crossentropy(model.out, label))
grads = theano.grad(cost, model.params)
updates = adadelta(model.params, grads)

f_train = theano.function(inputs=[sent,ques,label], outputs=cost, updates=updates, allow_input_downcast=True)


print('Training')
for n in range(EPOCHS):
    nb_batchs = X.shape[0]/BATCH_SIZE
    for i in range(nb_batchs):
        batch_X = X[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        batch_Xq = Xq[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        batch_Y = Y[i*BATCH_SIZE:i*BATCH_SIZE+BATCH_SIZE]
        cost = f_train(batch_X.T, batch_Xq.T, batch_Y)
        print(cost)