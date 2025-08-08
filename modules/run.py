from common import config
config.GPU = True
#from common.trainer import RnnlmTrainer
from trainer import trainer2
from trainer import trainer3
from SimpleRnnlm import SimpleRnnlm
from BetterRnnlm import BetterRnnlm
from common.optimizer import SGD
from dataset import ptb
from common.np import *
from common.util import eval_perplexity
from common.trainer import Trainer
from trainer import RnnlmTrainer
from LSTM_LM import Rnnlm


# data load
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_test, word_to_id_test, id_to_word_test = ptb.load_data('test')
#corpus_size = len(word_to_id)#1000
#corpus_size = len(word_to_id)
corpus_size = len(corpus)
corpus = corpus[:corpus_size]
#print(max(corpus))
#vocab_size = int(max(corpus) + 1)
vocab_size = int(len(word_to_id))

corpus = np.asarray(corpus)

xs = corpus[:-1]
ts = corpus[1:]
#print(xs)

# hyperparameters
# batch_size = 10
# wordvec_size = 100
# hidden_size = 100
# time_size = 5
# lr = 0.2
# max_epoch = 20
 

batch_size = 20
wordvec_size = 650
hidden_size = 650
time_size = 35
lr = 20.0
max_epoch = 40
max_grad = 0.25
dropout = 0.5


#xs = np.arange(1000)
model = BetterRnnlm(vocab_size, wordvec_size, hidden_size)
#model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)
total_loss = 0  
loss_count = 0
ppl_list = []

#max_grad = 0.75


# trainer = trainer2(model, optimizer)
# trainer.fit(xs, ts, corpus_size, max_epoch, batch_size, time_size, max_grad)



ppl_best = 100000
#trainer = RnnlmTrainer(model, optimizer)
trainer = trainer3(model, optimizer)
for _ in range(max_epoch):
    
    #trainer.fit(xs, ts, max_epoch, batch_size, time_size, max_grad)
    trainer.fit(xs, ts, corpus_size, max_epoch, batch_size, time_size, max_grad)

    model.reset_state()
    ppl_test = eval_perplexity(model, corpus_test, batch_size, time_size)
    if ppl_test < ppl_best:
        ppl_best = ppl_test
        model.save_params()
    else:
        lr /= 4.0
        optimizer.lr = lr
    print('Test PPL', ppl_test)
model.save_params()
model.reset_state()
#model.save_params()


