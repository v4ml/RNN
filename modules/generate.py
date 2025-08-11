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

    def generate2(self, start_id, skip_ids=None, sample_size=100):
        corpus, word_to_id, id_to_word = ptb.load_data('train')
        corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')
        vocab_size = int(len(word_to_id))

        start_id = word_to_id['you']
        word_ids = [start_id]

        skip_words = ['N', '<unk>', '$']
        skip_ids = [word_to_id[w] for w in skip_words]

        x = start_id
        # while len(word_ids) < sample_size:
        #     x = np.array(x).reshape(1, 1)
        #     score = self.predict(x).flatten()
        #     p = softmax(score).flatten()

        #     sampled = np.random.choice(len(p), size=1, p=p)
        #     if (skip_ids is None) or (sampled not in skip_ids):
        #         x = sampled
        #         word_ids.append(int(x))

        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            index = int(np.argmax(self.predict(x)))
            #score = self.predict(x).flatten()
            #p = softmax(score).flatten()

            #sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (index not in skip_ids):
                x = index
            word_ids.append(int(x))                

        #return word_ids
        
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


        self.reset_state()

        start_words = 'the meaning of life is'
        start_ids = [word_to_id[w] for w in start_words.split(' ')]

        for x in start_ids[:-1]:
            x = np.array(x).reshape(1, 1)
            self.predict(x)                
        word_ids = start_ids[:-1] + word_ids
        txt = ' '.join([id_to_word[i] for i in word_ids])
        txt = txt.replace(' <eos>', '.\n')
        print(txt)


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
gen.generate2('you')