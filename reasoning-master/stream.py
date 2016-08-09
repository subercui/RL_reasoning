import sys
import logging
logging.basicConfig(level=logging.INFO)
import cPickle as pickle
logger = logging.getLogger(__name__)
import numpy
import os
import configurations



config = getattr(configurations, 'get_config')()

def creat_dic_with_lines(lines):
    wordset = set(item for line in lines for item in line.strip().split())
    word2index = {word: index for index, word in enumerate(wordset)}
    return word2index

def process_line(line, add_head_tail=True):
    line = line.strip()
    if line.endswith(" |||"):
        line = line[:-4]
    line = line.replace('.', ' .')
    line = line.replace('?', ' ?')
    line = line.lower()
    line_vec = line.strip().split(" ||| ")
    if add_head_tail:
        line_vec = ['<s> {} </s>'.format(item) for item in line_vec]
    return line_vec

class preprocess(object):
    def __init__(self, fact_filename, question_filename, res_filename, dict_file=None):
        self.fact_filename = fact_filename
        self.question_filename = question_filename
        self.res_filename = res_filename
        if dict_file is None:
            self.dict_file = config['dict_file']

        self.vocab_dic, self.res_dic = self.creat_dic()


    @staticmethod
    def _creat_dict_with_files(filenames, **kwargs):
        lines = []
        for filename in filenames:
            f_handle = open(filename)
            for line in f_handle:
                line_vec = process_line(line, **kwargs)
                lines.extend(line_vec)
            f_handle.close()
        return creat_dic_with_lines(lines)

    def creat_dic(self):
        dic_file_name = self.dict_file
        if os.path.isfile(dic_file_name):
            dicfile = open(dic_file_name)
            return pickle.load(dicfile)

        vocab_dic = self._creat_dict_with_files([self.fact_filename,
                                                 self.question_filename])
        res_dic = self._creat_dict_with_files([self.res_filename], add_head_tail=False)
        vocab_dic['<unk>'] = len(vocab_dic)
        pickle.dump((vocab_dic, res_dic), open(dic_file_name, 'w'))
        logger.info('dict file dumped')
        logger.info('vocab size is {}'.format(len(vocab_dic)))
        logger.info('label size is {}'.format(len(res_dic)))
        return vocab_dic, res_dic

    @staticmethod
    def _mapline(s, word2index, **kwargs):
        s_vec = process_line(s, **kwargs)
        line_process = []
        ret_res = []
        for line in s_vec:
            line_vec = line.split()
            if '<unk>' in word2index:
                unkid = word2index['<unk>']
                line_process = [word2index.get(item, unkid) for item in line_vec]
            else:
                line_process = [word2index.get(item) for item in line_vec]
            ret_res.append(line_process)
        return ret_res

    def data_stream(self):
        fact_file = open(self.fact_filename)
        question_file = open(self.question_filename)
        res_file = open(self.res_filename)

        for fact_line, question_line, res_line in \
        zip(fact_file, question_file, res_file):
            fact_line = self._mapline(fact_line, self.vocab_dic)
            question_line = self._mapline(question_line, self.vocab_dic)
            res_line = self._mapline(res_line, self.res_dic, add_head_tail=False)
            data =  self.block_padding(fact_line), \
                self.block_padding(question_line), \
                numpy.asarray(res_line, dtype='int64').flatten()
            yield data

    @staticmethod
    def block_padding(sentences):
        batch_size = len(sentences)
        width = max([len(item) for item in sentences])
        block = numpy.zeros((batch_size, width), dtype='int64')
        padding = numpy.zeros((batch_size, width), dtype='float32')
        for index in range(batch_size):
            lens = len(sentences[index])
            block[index, :lens] = sentences[index]
            padding[index, :lens] = 1.
        return block, padding



if __name__ == '__main__':
    train_file = config['train_file']
    data_class = preprocess(*train_file)
    # for data in data_class.data_stream():
        # print data[2]
        # break


