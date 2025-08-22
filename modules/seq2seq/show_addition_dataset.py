# coding: utf-8
import sys

from numpy import float32

#from modules.generate_better_text import vocab_size
sys.path.append('./modules')
from dataset import sequence
from encoder import Encoder
from encoder import Decoder
from common.np import *

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
N, T1 = t_train.shape
batch = N//batch_size

encoder = Encoder(vocab_size, wordvec_size, hidden_size)
hs = np.empty((N, hidden_size), dtype=float32)
#ts = np.argmax(t_train, axis=-1)

for i in range(batch):
    h = encoder.forward(x_train[batch_size*i:batch_size*(i+1), :], t_train[batch_size*i:batch_size*(i+1), :])    
    hs[batch_size*i:batch_size*(i+1), :] = h

decoder = Decoder(vocab_size, wordvec_size, hidden_size)
xs = np.full((20,1), 6)
for i in range(batch):
    for i in range(T1):
        decoder.forward(xs , t_train[batch_size*i:batch_size*(i+1), :], h)


N = 20
T = 7
H = 10

#for n in Range(N):

