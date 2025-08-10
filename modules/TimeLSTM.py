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
            layer = LSTM2(Wx, Wh, b)
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
    
class TimeLSTM3:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful
        
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None    