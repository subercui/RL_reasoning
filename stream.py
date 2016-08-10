import sys
import logging
logging.basicConfig(level=logging.INFO)
import cPickle as pickle
logger = logging.getlogger(__name__)
import numpy
import os
import configurations


config = getattr(configurations, 'get_config')()

def creat_dic_with_lines(lines):
	wordset = set(item for line in lines for item in line.strip().split())
	word2index = {word: index for index, word in enumerate(wordset)}
	return word2index
	

#whats this doing?	
def process_line(line, add_head_tail=True):
	line =line.strip()
	if line.endswith(" |||"):
		line = line[:-4]
	line = line.replace('.', ' .')
	line = line.replace('?', ' ?')
	line = line.lower()
	line_vec = line.strip().split(" ||| ")
	if add_head_tail:
		line_vec = ['<s>{}</s>'.format(item) for item in line_vec]
	return line_vec
	
class preprocess(object):
	def __init__(self, fact_filename, question_filename, res_filename, dict_file=None):
        self.fact_filename = fact_filename
        self.question_filename = question_filename
        self.res_filename = res_filename
        if dict_file is None:
            self.dict_file = config['dict_file']

        self.vocab_dic, self.res_dic = self.creat_dic()
