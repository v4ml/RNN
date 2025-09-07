import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BetterRnnlm import BetterRnnlm
from BetterRnnlm import BetterRnnlm2
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

class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        V,D,H = vocab_size, wordvec_size, hidden_size
        #self.params = [N, T, V, D, H]
        self.rnnlm = BetterRnnlm2(vocab_size, wordvec_size, hidden_size)
        #self.rnnlm.load_params('./BetternRnnlm.pkl')

        rn = np.random.randn
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
            #dropout(dropout_ratio),
            #TimeLSTM(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            #TimeAffine2(embed_W.T, affine_b)
            #TimeAffine(affine_W, affine_b)
            #TimeAffine_nob(affine_W)
        ]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs):
        N, T = xs.shape
        for layer in self.layers:
            xs = layer.forward(xs)
        #hs = xs[:, T-1, :]
        return xs

    def backward(self, dout):
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size, dropout_ratio=0.5):
        H = hidden_size
        V = vocab_size
        D = wordvec_size

        rn = np.random.randn
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
        affine_W = (rn(H,V)/np.sqrt(H)).astype('f')
        affine_b = np.zeros(vocab_size).astype('f')

        self.layers = [
            TimeEmbedding2(embed_W),
            dropout(dropout_ratio),            
            #TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeLSTM2(rnn_Wx0, rnn_Wh0, rnn_b0, stateful=True),
            dropout(dropout_ratio),
            TimeLSTM2(rnn_Wx1, rnn_Wh1, rnn_b1, stateful=True),
            #dropout(dropout_ratio),
            #TimeLSTM(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine2(affine_W, affine_b)
            #TimeAffine(affine_W, affine_b)
            #TimeAffine_nob(affine_W)
        ]
        #self.affine_layer = TimeAffine2(affine_W, affine_b)
        self.loss_layer = TimeSoftmaxWithLoss3()
        self.rnn_layer = [self.layers[2], self.layers[4]]

        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            

    def forward(self, xs, ts, prev_h, generate=False):
        N, T = ts.shape
        N, D = prev_h.shape

        self.layers[2].set_state(prev_h)
        xss = np.empty((N,T,D), dtype=xs.dtype)
        hss = np.empty((N,T,D), dtype=xs.dtype)
        xts = np.empty((N, 1), dtype='int')
        #xs = np.full((20,1), 6)

        for layer in self.layers:
            xs = layer.forward(xs)

        self.cache = [xs]  
        
        loss = self.loss_layer.forward(xs, ts)
        return loss
    
    def backward(self, dout=1):
        
        dout = self.loss_layer.backward(dout)
        dh = dout.copy()
        N,T,H = dout.shape
        for t in range(len(self.layers)-1, 0, -1):
            dh = self.layers[t].backward(dh)
        dout = self.layers[0].backward(dh)
        dh = np.sum(dh, axis=1)
        return dout, dh
    
    def generate(self, start_id, length, prev_h):
        #T = length
        xs = self.cache[0]
        N, T, V = xs.shape
        N, H = prev_h.shape

        self.layers[2].set_state(prev_h)
        xss = np.zeros((1,V), dtype='int')
        hss = npxzzzzzzzzzzzzzz xdxxs.empty((1,T,H), dtype=xs.dtype)
        xts = np.empty((1, 1), dtype='int')
        #xs = np.full((20,1), 6)

        for layer in self.layers:
            xss = layer.forward(xss)

        self.cache = [xs]  


    def forward2(self, ts, last_h):
        N, T = ts.shape
        N, D = last_h.shape

        self.layers[2].set_state(last_h)
        xss = np.empty((N,T,D), dtype=xs.dtype)
        hss = np.empty((N,T,D), dtype=xs.dtype)
        xts = np.empty((N, 1), dtype='int')
        xs = np.full((20,1), 6)
        for t in range(T):
            for layer in self.layers:
                xs = layer.forward(xs)
        
            hss[:, t, :] = xs[:, 0, :]
            xs = xs[:, 0, :].argmax(axis=1).reshape(20,1)
            xss[:, t, :] = xs[:, 0, :]
        hss = self.affine_layer.forward(hss)
            #hss[:, t, :] = hs[:, 0, :]
            # 다음 단어 임베딩 추출
        
        self.cache = [xss]  
            


        loss = self.loss_layer.forward(xss, ts)
        return loss
    
    
        

    
    def backward2(self, dout=1):
        
        dout = self.loss_layer.backward(dout)
        
        dout = self.affine_layer.backward(dout)
        xss = self.cache[0]
        N,T,H = dout.shape
        #for layer in reversed(self.layers):
        douts = np.zeros(dout.shape, dtype=dout.dtype)
        for t in range(T-1, -1, -1):
            dout = self.layers[4].backward(dout[:, 0, :].reshape(N,1,H))
            douts [:, t, :] = dout[:, 0, :]
        dout = self.layers[3].backward(douts)
        for t in range(T-1, -1, -1):
            dout = self.layers[2].backward(dout[:, 0, :].reshape(N,1,H))
            douts [:, t, :] = dout[:, 0, :]
        dout = self.layers[1].backward(douts)
        dout = self.layers[0].backward(dout)
        
        return dout

        
    


