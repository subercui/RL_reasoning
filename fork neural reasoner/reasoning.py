from model import auto_encoder, LogisticRegression, lookup_table
import theano
import theano.tensor as T
from utils import param_init, repeat_x


class Reason(object):

    def __init__(self, **kwargs):
        self.vocab_size = kwargs.pop('vocab_size')
        self.n_in = kwargs.pop('nemb')
        self.n_hids = kwargs.pop('nhids')
        self.n_layer = kwargs.pop('n_layer')
        self.n_label = kwargs.pop('label_size')
        self.params=[]

    def apply(self, facts, facts_mask, question, question_mask, y):

        table = lookup_table(self.n_in, self.vocab_size)
        self.params += table.params

        facts_encoder = auto_encoder(facts, facts_mask, self.vocab_size,
                                     self.n_in, self.n_hids, table=table)

        questions_encoder = auto_encoder(question, question_mask, self.vocab_size,
                                         self.n_in, self.n_hids, table=table)

        self.params += facts_encoder.params
        self.params += questions_encoder.params

        facts_rep = facts_encoder.output
        questions_rep = questions_encoder.output

        for _ in range(self.n_layer):
            questions_rep = self.interact(facts_rep, questions_rep)

        logistic_layer = LogisticRegression(questions_rep,
                                            self.n_hids, self.n_label)
        self.params += logistic_layer.params
        self.cost = logistic_layer.cost(y)/y.shape[0]
        self.decoder_cost = facts_encoder.cost + questions_encoder.cost

        self.errors = logistic_layer.errors(y)
        return self.cost, self.decoder_cost

    def interact(self, facts_rep, questions_rep):
        self.W_f = param_init().orth((self.n_hids, self.n_hids))
        self.W_q = param_init().orth((self.n_hids, self.n_hids))
        self.b_f = param_init().constant((self.n_hids,))
        self.b_q = param_init().constant((self.n_hids,))
        self.params += [self.W_f, self.W_q, self.b_f, self.b_q]

        questions_rep = T.tanh(theano.dot(questions_rep, self.W_q) + self.b_q)
        facts_rep = T.tanh(theano.dot(facts_rep, self.W_f) + self.b_f)

        def _one_step(question_rep, facts_rep):
            if question_rep.ndim == 1:
                question_rep = T.shape_padleft(question_rep, n_ones=1)
            inter_rep = (question_rep + facts_rep).max(axis=0)
            return inter_rep

        inter_reps, updates = theano.scan(_one_step,
                                          sequences=questions_rep,
                                          outputs_info=None,
                                          non_sequences=facts_rep
                                          )
        return inter_reps

