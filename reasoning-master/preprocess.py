import configurations

def pre_procsee(f_name, q_name, l_name, r_name):

    facts = open(f_name, 'w')
    question = open(q_name, 'w')
    answer = open(l_name, 'w')
    source = open(r_name, 'r')

    Fact_type, Question_type = 0, 1
    def is_question(s_):
        s_v = s_.strip().split()
        if s_v[0].isdigit() and s_v[-1].isdigit() and s_.find('?') != -1:
            return Question_type
        else:
            return Fact_type

    pre_type = Fact_type
    Question_num = 0
    for line in source:
        line = line.strip()
        cur_type = is_question(line)
        if cur_type == Fact_type:
            line = " ".join(line.split()[1:])
            if pre_type == Fact_type:
                facts.write(line + " ||| ")
            else:
                facts.write('\n' + line + ' ||| ')
        else:
            Question_num += 1
            index = line.index('?')
            part1 = line[:index+1]
            part2 = line[index+1:]
            q = " ".join(part1.split()[1:])
            a = part2.strip().split()[0]
            if pre_type == Question_type or Question_num == 1:
                question.write(q+" ||| ")
                answer.write(a+" ||| ")
            else:
                question.write('\n' + q+" ||| ")
                answer.write('\n' + a+" ||| ")

        pre_type = cur_type

    answer.write('\n')
    facts.write('\n')
    question.write('\n')

if __name__=='__main__':
    config = getattr(configurations, 'get_config')()
    train_args = config['train_file'] + [config['raw_train']]
    test_args = config['test_file'] + [config['raw_test']]
    print train_args
    print test_args
    pre_procsee(*train_args)
    pre_procsee(*test_args)

