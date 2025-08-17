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
from Dropout import dropout2
from Dropout import dropout
from common.time_layers import TimeLSTM
import pickle

class BetterRnnlm:
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        N = 10
        T = 5
        V,D,H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        #가중치 초기화
        embed_W = (rn(V,D)/100).astype('f')
        rnn_Wx0 = (rn(D,H*4)/np.sqrt(D)).astype('f')
        rnn_Wh0 = (rn(H,H*4)/np.sqrt(H)).astype('f')
        rnn_b0 = np.zeros(H*4).astype('f')
        rnn_Wx1 = (rn(H,H*4)/np.sqrt(H)).astype('f')
        rnn_Wh1= (rn(H,H*4)/np.sqrt(H)).astype('f')
        rnn_b1 = np.zeros(H*4).astype('f')
        #affine_W = (rn(N,T,H,V)/np.sqrt(H)).astype('f')
        #affine_W = np.random.randn(1,T,H,V)
        #affine_b = np.zeros((N,T), dtype='f')
        #affine_W = (embed_W.T/np.sqrt(H)).astype('f')
        #affine_W = (rn(H,V)/np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.layers = [
            TimeEmbedding2(embed_W),
            dropout(dropout_ratio),
            #TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeLSTM2(rnn_Wx0, rnn_Wh0, rnn_b0, stateful=True),
            dropout(dropout_ratio),
            TimeLSTM2(rnn_Wx1, rnn_Wh1, rnn_b1, stateful=True),
            dropout(dropout_ratio),
            #TimeLSTM(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine2(embed_W.T, affine_b)
            #TimeAffine(affine_W, affine_b)
            #TimeAffine_nob(affine_W)
        ]


        # self.load_params('./BetterRnnlm.pkl')
        # self.layers = [
        #     TimeEmbedding2(self.params[0]),
        #     #dropout(dropout_ratio),
        #     #TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
        #     TimeLSTM2(self.params[1], self.params[2], self.params[3], stateful=True),
        #     #dropout(dropout_ratio),
        #     TimeLSTM2(self.params[4], self.params[5], self.params[6], stateful=True),
        #     #dropout(dropout_ratio),
        #     #TimeLSTM(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
        #     TimeAffine2(self.params[7], self.params[8])
        #     #TimeAffine(affine_W, affine_b)
        #     #TimeAffine_nob(affine_W)
        # ]
        self.loss_layer = TimeSoftmaxWithLoss3()
        self.rnn_layer = [self.layers[2], self.layers[4]]

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
        for i in range(len(self.rnn_layer)):
            self.rnn_layer[i].reset_state()

    def save_params(self, file_name='rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name='./modules/dataset/20250806.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)
        self.params = [np.asarray(p) for p in self.params]


class BetterRnnlm2:
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        V,D,H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        #가중치 초기화
        embed_W = (rn(V,D)/100).astype('f')
        rnn_Wx0 = (rn(D,H*4)/np.sqrt(D)).astype('f')
        rnn_Wh0 = (rn(H,H*4)/np.sqrt(H)).astype('f')
        rnn_b0 = np.zeros(H*4).astype('f')
        rnn_Wx1 = (rn(H,H*4)/np.sqrt(H)).astype('f')
        rnn_Wh1= (rn(H,H*4)/np.sqrt(H)).astype('f')
        rnn_b1 = np.zeros(H*4).astype('f')
        #affine_W = (rn(N,T,H,V)/np.sqrt(H)).astype('f')
        #affine_W = np.random.randn(1,T,H,V)
        #affine_b = np.zeros((N,T), dtype='f')
        #affine_W = (embed_W.T/np.sqrt(H)).astype('f')
        affine_W = (rn(H,5)/np.sqrt(H)).astype('f')
        affine_b = np.zeros(5).astype('f')

        self.layers = [
            TimeEmbedding2(embed_W),
            dropout(dropout_ratio),
            #TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeLSTM2(rnn_Wx0, rnn_Wh0, rnn_b0, stateful=True),
            dropout(dropout_ratio),
            TimeLSTM2(rnn_Wx1, rnn_Wh1, rnn_b1, stateful=True),
            dropout(dropout_ratio),
            #TimeLSTM(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine2(affine_W, affine_b)
            #TimeAffine(affine_W, affine_b)
            #TimeAffine_nob(affine_W)
        ]


        # self.load_params('./BetterRnnlm.pkl')
        # self.layers = [
        #     TimeEmbedding2(self.params[0]),
        #     #dropout(dropout_ratio),
        #     #TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
        #     TimeLSTM2(self.params[1], self.params[2], self.params[3], stateful=True),
        #     #dropout(dropout_ratio),
        #     TimeLSTM2(self.params[4], self.params[5], self.params[6], stateful=True),
        #     #dropout(dropout_ratio),
        #     #TimeLSTM(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
        #     TimeAffine2(self.params[7], self.params[8])
        #     #TimeAffine(affine_W, affine_b)
        #     #TimeAffine_nob(affine_W)
        # ]
        self.loss_layer = TimeSoftmaxWithLoss3()
        self.rnn_layer = [self.layers[2], self.layers[4]]

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
        for i in range(len(self.rnn_layer)):
            self.rnn_layer[i].reset_state()

    def save_params(self, file_name='rnnlm.pkl'):
        with open(file_name, 'wb') as f:
            pickle.dump(self.params, f)

    def load_params(self, file_name='./modules/dataset/20250806.pkl'):
        with open(file_name, 'rb') as f:
            self.params = pickle.load(f)
        self.params = [np.asarray(p) for p in self.params]        