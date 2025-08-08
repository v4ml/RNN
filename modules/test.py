import sys
import time
sys.path.append('..')
#import matplotlib.pyplot as plt
from dataset import ptb
#import numpy as np
from common.np import *
from SimpleRnnlm import SimpleRnnlm

from common.optimizer import SGD
from util import clip_grads
import matplotlib.pyplot as plt
from LSTM_LM import Rnnlm


# data load
corpus, word_to_id, id_to_word = ptb.load_data('train')
#corpus_size = len(word_to_id)#1000
corpus_size = len(word_to_id)
#corpus_size = len(corpus)
corpus = corpus[:corpus_size]

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
# vocab_size = int(max(corpus) + 1)


batch_size = 20
wordvec_size = 100
hidden_size = 100
time_size = 35
lr = 20.0
max_epoch = 1
vocab_size = int(len(word_to_id) + 1)


#xs = np.arange(1000)
offset = 0
max_iters = len(xs) // (batch_size * time_size)
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
#model = Rnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

total_loss = 0
loss_count = 0
ppl_list = []

for epoch in range(max_epoch):
    temp = corpus_size//batch_size
    for iters in range(max_iters):
        idx = np.empty((batch_size, time_size), dtype='i')
        x = [] #np.arange(batch_size, 5)
        t = []
        #temp = np.arange(5)
        
        for batch_idx in range(batch_size):
            start_idx = ((batch_idx*time_size)+offset)%(corpus_size-1)
            #idx[batch_idx] = [(i%(corpus_size-1)) for i  in range(start_idx, start_idx+time_size)]
            idx[batch_idx] = np.array([(i % (corpus_size - 1)) for i in range(start_idx, start_idx + time_size)])

            #print(idx[batch_idx])
            #print(xs[idx[batch_idx]])
            #temp = xs[idx[batch_idx]]
            x.append(xs[idx[batch_idx]])
            t.append(ts[idx[batch_idx]])

        if(iters==1325):
            a=1

        offset = offset+time_size
        x = np.array(x)
        t = np.array(t)
        #print(x)
        loss = model.forward(x, t)
        #print(model.grads[1].__array_interface__['data'])
        model.backward()
        #print(model.grads[1].__array_interface__['data'])
        clip_grads(model.grads, 0.25)

        optimizer.update(model.params, model.grads)


        #print('learning : %d' % model.grads[2].__array_interface__['data'][0])
        #print('loss %d' % (loss))
        total_loss += loss
        loss_count += 1

        #print(x)
        #print("=========== iter end===")
        ppl = np.exp(total_loss / loss_count)
        print('| Epoch %d | PPL %.2f' % (epoch+1, ppl))
        ppl_list.append(float(ppl))
        total_loss, loss_count = 0, 0

ylim = None
x = np.arange(len(ppl_list))
if ylim is not None:
    plt.ylim(*ylim)
plt.plot(x.get(), ppl_list, label='train')
plt.xlabel('반복 (x' + str(20) + ')')
plt.ylabel('손실')
plt.show()
    #print("=========== "+str(epoch)+" epoch end===================")