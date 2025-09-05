# coding: utf-8
import sys

from numpy import float32

#from modules.generate_better_text import vocab_size
sys.path.append('./modules')
from dataset import sequence
from encoder import Encoder
from encoder import Decoder
from common.np import *
from seq2seq import Seq2seq

(x_train, t_train), (x_test, t_test) = \
    sequence.load_data('addition.txt', seed=1984)
char_to_id, id_to_char = sequence.get_vocab()

print(x_train.shape, t_train.shape)
print(x_test.shape, t_test.shape)
# (45000, 7) (45000, 5)
# (5000, 7) (5000, 5)

print(x_train[0])
print(t_train[0])
# [ 3  0  2  0  0 11  5]
# [ 6  0 11  7  5]

print(''.join([id_to_char[c] for c in np.asnumpy(x_train[0])]))
print(''.join([id_to_char[c] for c in np.asnumpy(t_train[0])]))

x_train = x_train[:1000, :]
t_train = t_train[:1000, :]
# 71+118
# _189

vocab_size = len(char_to_id)
wordvec_size = 15
hidden_size = 15
time_size = 7
batch_size = 20 
N, T = x_train.shape
N, Tt = t_train.shape
batch = N//batch_size

encoder = Encoder(vocab_size, wordvec_size, hidden_size)
hs = np.empty((N, hidden_size), dtype=float32)

decoder = Decoder(vocab_size, wordvec_size, hidden_size)
#ts = np.argmax(t_train, axis=-1)

#for i in range(batch):

tt_train = np.zeros_like(t_train, dtype=np.int32) #np.zeros((N,Tt), dtype=t_train.dtype)
tt_train[:, 4] = 5
print("t_train.shape :  ", t_train.shape)
print("tt_train.shape :  ", tt_train.shape)
tt_train[:, :4] = np.array(t_train[:, 1:5], dtype=tt_train.dtype)

max_epoch = 50

model = Seq2seq(encoder, decoder, batch_size, hidden_size, time_size)

for epoch in range(1):
    model.forward(x_train, t_train)
    model.backward()


    model.generate(x_test, char_to_id.get('_'), 5)

    # for i in range(batch):
    #     hs = encoder.forward(x_train[batch_size*i:batch_size*(i+1), :], t_train[batch_size*i:batch_size*(i+1), :])    
    #     #hs[batch_size*i:batch_size*(i+1), :] = h
        
    #     #for i in range(T1):


    #     loss = decoder.forward(t_train[batch_size*i:batch_size*(i+1), :], tt_train[batch_size*i:batch_size*(i+1), :], hs[:, -1])
    #     dout, dh = decoder.backward()
    #     dh = dh.reshape(batch_size,1,hidden_size)
    #     dh = np.repeat(dh, 7, axis=1)
    #     dout = encoder.backward(dh)
    



def eval_perplexity(model, corpus, batch_size=10, time_size=35):
    print('퍼플렉서티 평가 중 ...')
    corpus_size = len(corpus)
    total_loss, loss_cnt = 0, 0
    max_iters = (corpus_size - 1) // (batch_size * time_size)
    jump = (corpus_size - 1) // batch_size

    for iters in range(max_iters):
        xs = np.zeros((batch_size, time_size), dtype=np.int32)
        ts = np.zeros((batch_size, time_size), dtype=np.int32)
        time_offset = iters * time_size
        offsets = [time_offset + (i * jump) for i in range(batch_size)]
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                xs[i, t] = corpus[(offset + t) % corpus_size]
                ts[i, t] = corpus[(offset + t + 1) % corpus_size]

        try:
            loss = model.forward(xs, ts, train_flg=False)
        except TypeError:
            loss = model.forward(xs, ts)
        total_loss += loss

        sys.stdout.write('\r%d / %d' % (iters, max_iters))
        sys.stdout.flush()

    print('')
    ppl = np.exp(total_loss / max_iters)
    return ppl

N = 20
T = 7
H = 10

#for n in Range(N):

