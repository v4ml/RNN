from BetterRnnlm import BetterRnnlm
from dataset import ptb
from common.optimizer import SGD
from common.np import *
from common import config
config.GPU = True
# commit TEST22
class RnnlmGen(BetterRnnlm):
    # def __init__(self, model):
    #     self.model = model

    def generate2(self, start_word, skip_ids=None, sample_size=1000):
        corpus, word_to_id, id_to_word = ptb.load_data('train')
        corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')
        vocab_size = int(len(word_to_id))

        skip_words = ['N', '<unk>', '$']
        skip_ids = [word_to_id[w] for w in skip_words]

        xs = word_to_id[start_word]
        
        #xs = np.asarray(xs)
        word_ids = [word_to_id[start_word]]

        wordvec_size = 650
        hidden_size = 650
        lr = 20.0

        model = BetterRnnlm(vocab_size, wordvec_size, hidden_size)
        model.load_params()

        while len(word_ids) < sample_size:
            xs = np.array(xs).reshape(1, 1) 
            #index = int(np.argmax(model.predict(xs)))
            score = self.predict(xs).flatten()
            p = self.softmax(score).flatten()
            index = np.random.choice(len(p), size=1, p=p)

            if (skip_ids is None) or (index not in skip_ids):
                xs = index
                word_ids.append(int(index))
        
        sentence = ' '.join([id_to_word[i] for i in word_ids])
        print(sentence)

    def generate(self, start_id, skip_ids=None, sample_size=1000):
        corpus, word_to_id, id_to_word = ptb.load_data('train')
        corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')
        vocab_size = int(len(word_to_id))
        
        skip_words = ['N', '<unk>', '$']
        skip_ids = [word_to_id[w] for w in skip_words]

        x = word_to_id['you']
        word_ids = [x]
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1) 
            score = self.predict(x).flatten()
            #p = self.softmax(score).flatten()

            #sampled = np.random.choice(len(p), size=1, p=p)
            index = int(np.argmax(score))
            if (skip_ids is None) or (index not in skip_ids):
                x = index
                word_ids.append(int(x))

        sentence = ' '.join([id_to_word[i] for i in word_ids])
        print(sentence)

    def softmax(self, x):
        if x.ndim == 2:
            x = x - x.max(axis=1, keepdims=True)
            x = np.exp(x)
            x /= x.sum(axis=1, keepdims=True)
        elif x.ndim == 1:
            x = x - np.max(x)
            x = np.exp(x) / np.sum(np.exp(x))

        return x        


gen = RnnlmGen(10000, 650, 650)
gen.generate('you')