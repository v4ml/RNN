import sys
sys.path.append('..')
#import numpy as np
from common.np import *
from common.time_layers import TimeEmbedding
from TimeEmbedding import TimeEmbedding2
#from common.time_layers import TimeRNN
from TimeRNN import TimeRNN
#from common.time_layers import TimeAffine_nob
from common.time_layers import TimeAffine
from Affine import TimeAffine2
from common.time_layers import TimeSoftmaxWithLoss
from TimeSoftmaxWithLoss import TimeSoftmaxWithLoss2
from TimeSoftmaxWithLoss import TimeSoftmaxWithLoss3
#from LSTM import LSTM
from TimeLSTM import TimeLSTM2
from common.time_layers import TimeLSTM
import pickle

class SimpleRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        N = 10
        T = 5
        V,D,H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        #가중치 초기화
        embed_W = (rn(V,D)/100).astype('f')
        rnn_Wx = (rn(D,H*4)/np.sqrt(H)).astype('f')
        rnn_Wh = (rn(H,H*4)/np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H*4).astype('f')
        #affine_W = (rn(N,T,H,V)/np.sqrt(H)).astype('f')
        #affine_W = np.random.randn(1,T,H,V)
        #affine_b = np.zeros((N,T), dtype='f')
        affine_W = (rn(H,V)/np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        # 계층
        self.layers = [
            TimeEmbedding(embed_W),
            #TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeLSTM2(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            #TimeLSTM(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
            #TimeAffine(affine_W, affine_b)
            #TimeAffine_nob(affine_W)
        ]
        self.loss_layer = TimeSoftmaxWithLoss3()
        self.rnn_layer = self.layers[1]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss
        
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
        
    def reset_state(self):
        self.rnn_layer.reset_state()

    def save_params(self, file_name='rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name='rnnlm.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)