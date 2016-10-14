echo $$
THEANO_FLAGS='floatX=float32' python main.py 0.9 >log1 2>&1 &
THEANO_FLAGS='floatX=float32' python main.py 0.85 >log2 2>&1 &
wait
echo 'end'
