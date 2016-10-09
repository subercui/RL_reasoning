# import packages/functions
from theano import shared, scan, function, tensor as T
import numpy as np

# declare variables
X = T.dmatrix("X")
Wx = shared(np.random.uniform(-1.0, 1.0, (10, 20)))
Wh = shared(np.random.uniform(-1.0, 1.0, (20, 20)))
b = shared(np.random.uniform(-1.0, 1.0, (1, 20)))

# define recurrence function
def recurrence(x_t, h_tm1):
    return T.nnet.sigmoid(T.dot(h_tm1, Wh) + T.dot(x_t, Wx) + b)

# compute hidden state sequence with scan
ht, _ = scan(fn = recurrence, sequences = X,
             outputs_info = np.zeros((1, 20)))

# define function producing hidden state sequence
fn = function([X], ht)

# test function
print fn(np.eye(10))

"""
the code from:
http://stackoverflow.com/questions/33696775/why-does-this-minimal-rnn-code-throw-a-type-error-for-a-type-never-used


Using the edited code, and the latest version of Theano, I get the error

TypeError: ('The following error happened while compiling the node', forall_inplace,cpu,scan_fn}(Shape_i{0}.0, Subtensor{int64:int64:int8}.0, IncSubtensor{InplaceSet;:int64:}.0, , , ), '\n', "Inconsistency in the inner graph of scan 'scan_fn' : an input and an output are associated with the same recurrent state and should have the same type but have type 'TensorType(float64, row)' and 'TensorType(float64, matrix)' respectively.")
This is similar in spirit to the question's error, but refers to a mismatch between a matrix and row vector; there is no reference to a 3-tensor.

The simplest change to avoid this error is to change the shape of the b shared variable.

Instead of

b = shared(np.random.uniform(-1.0, 1.0, (1, 20)))
use

b = shared(np.random.uniform(-1.0, 1.0, (20,)))
I would also recommend doing the same for the initial value of h_tm1.

Instead of

outputs_info = np.zeros((1, 20))
use

outputs_info = np.zeros((20,))

"""

"""
It seems that
- if output_info is like layout(1,20), meaning the first dim is 1, it will pass a TensorType(float64, row) to the inner function
  of scan.
- else if output_info is like layout(3,20), it pass a TensorType(float64, matrix) to the inner function

So my solution is recorrect it to outputs_info = T.unbroadcast(T.zeros((1,20)), 0)), and this make sure passing a
TensorType(float64, matrix) to the inner function.

The blow code is my runnable version
"""

# import packages/functions
from theano import shared, scan, function, config, tensor as T
import numpy as np

# declare variables
X = T.tensor3("X")
Wx = shared(np.asarray(np.random.uniform(-1.0, 1.0, (10, 20)), dtype=config.floatX))
Wh = shared(np.asarray(np.random.uniform(-1.0, 1.0, (20, 20)), dtype=config.floatX))
b = shared(np.asarray(np.random.uniform(-1.0, 1.0, (1, 20)), dtype=config.floatX))

# define recurrence function
def recurrence(x_t, h_tm1):
    print h_tm1.type, x_t.type
    return T.nnet.sigmoid(T.dot(h_tm1, Wh) + T.dot(x_t, Wx) + b)

# compute hidden state sequence with scan
ht, _ = scan(fn = recurrence, sequences = X,
             outputs_info = T.unbroadcast(T.zeros((1,20)), 0))

# define function producing hidden state sequence
fn = function([X], ht)

# test function
print fn(np.eye(10,10,dtype=config.floatX).reshape(10,1,10))