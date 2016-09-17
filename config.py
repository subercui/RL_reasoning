def get_config():


	config = {}
	config['nhids'] = 39 #hidden units
	config['nemb'] = 30 #
	config['n_layer'] = 3
	#the following two is set by the data, when run the stream.py
	config['vocab_size'] = 24
	config['label_size'] = 12 
	
	datadir ='/home/lxg/workspace-python/RL_reasoning/data/'
	
	### raw training data and test data ###
	raw_train_file = datadir + 'qa19_path-finding_train.txt'
	raw_test_file = datadir + 'qa19_path-finding_test.txt'
	config['raw_train'] = raw_train_file
	config['raw_test'] = raw_test_file
	
	
	#######temp data
	tmp_suffix = ['.fact', '.ques', '.answer']
	head_suffix = datadir+"19"
	config['train_file'] = [head_suffix+".train"+suffix for suffix in tmp_suffix]
	config['test_file'] = [head_suffix+".test"+suffix for suffix in tmp_suffix]
	
	####### dict file
	config['dict_file'] = head_suffix +"pathfinding_19dict.pkl"

# for RL
	config['n_itr'] = 12
	config['n_eps'] = 0.4
	config['T'] = 10
	config['discount'] = 0.7
	config['final_award'] = 1
	config['stp_penalty'] = -0.1
	config['n_grus'] = 1
	config['n_hts'] = 10
	config['n_lhids'] = 10
	config['stp_thrd'] = 10





	
	return config
