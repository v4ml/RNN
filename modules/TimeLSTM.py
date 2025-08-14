#import numpy as np
from common.np import *
from LSTM import LSTM
from LSTM import LSTM2

class TimeLSTM2:
    def __init__(self, Wx, Wh, b, stateful=False):
        #Wx = np.tile(Wx, (1,4))
        #Wh = np.tile(Wh, (1,4))
        #b = np.tile(b, 4)
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        self.stateful = stateful
        self.h, self.c = None, None
        

    def forward(self, xs):
        self.layers =[]
        Wx, Wh, b = self.params
        #T, H = Wh.shape
        H = Wh.shape[0]
        N, T, D = xs.shape

        if self.h is None or not self.stateful:
            self.h = np.zeros((N, H), dtype='f')
        if self.c is None or not self.stateful:            
            self.c = np.zeros((N, H), dtype='f')

        hs = np.zeros((N, T, H), dtype='f')
        #cs = np.zeros((N, T, H), dtype='f')
        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h
            #cs[:, t, :] = self.c
            self.layers.append(layer)

        #self.cache = [hs, cs]
        return hs
        
    
    def backward(self, dhs):
        Wx, Wh, b = self.params

        D = Wx.shape[0]
        N, T, H = dhs.shape
        #hs, cs = self.cache
        dxs = np.zeros((N, T, D), dtype='f')
        dh, dc = 0, 0
        #dc = np.zeros((N, H), dtype='f')
        #dh = np.zeros((N, H), dtype='f')
        
        grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        for t in reversed(range(T)):
            dxs[:, t, :], dh, dc = self.layers[t].backward(dhs[:, t, :]+dh, dc)
            grads[0] += self.layers[t].grads[0]
            grads[1] += self.layers[t].grads[1]
            grads[2] += self.layers[t].grads[2]

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None