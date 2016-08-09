def get_config():
    config = {}
    config['nhids'] = 30
    config['nemb'] = 30
    config['n_layer'] = 3
    config['vocab_size'] = 21
    config['label_size'] = 2

    #####################
    datadir = './data/'

    ### raw training data and test data
    raw_train_file = datadir + 'qa17_positional-reasoning_train.txt'
    raw_test_file = datadir + 'qa17_positional-reasoning_test.txt'
    config['raw_train'] = raw_train_file
    config['raw_test'] = raw_test_file

    ###### temp data
    tmp_suffix = ['.fact', '.ques', '.answer']
    head_suffix = datadir+"17"
    config['train_file'] = [head_suffix+".train"+suffix for suffix in tmp_suffix]
    config['test_file'] = [head_suffix+".test"+suffix for suffix in tmp_suffix]

    ###### dict file
    config['dict_file'] = head_suffix + "dict.pkl"


    return config

