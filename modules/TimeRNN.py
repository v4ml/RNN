from common.time_layers import RNN
import numpy as np

class TimeRNN:
    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        print('TIMERnn init : %d' % self.grads[1].__array_interface__['data'][0])
        self.h, self.dh = None, None
        self.stateful = stateful
        self.layers = []

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape
        self.layers = []

        if self.h is None or not self.stateful:
            self.h = np.zeros((N, H), dtype='f')
        
        hs = np.empty((N,T,H), dtype='f')
        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape
        dxs = np.empty((N,T,D), dtype='f')
        dh = np.zeros((N, H), dtype='f')

        for i in range(len(self.grads)):
            self.grads[i].fill(0)  # 반드시 초기화
        before = self.grads                    
        #self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        #print('backward() : %d' % self.grads[1].__array_interface__['data'][0])
        after = self.grads
        #print(before is after)
        
        for t in reversed(range(T)):
            dx, dh = self.layers[t].backward(dhs[:, t, :]+dh)
            dxs[:, t, :] = dx

            before = self.grads[2]
            self.grads[0] += self.layers[t].grads[0]
            self.grads[1] += self.layers[t].grads[1]
            self.grads[2] += self.layers[t].grads[2]
            after = self.grads[2]
            #print(before is after)
        #     for i, grad in enumerate(self.layers[t].grads):
        #         grads[i] += grad
                
        # for i, grad in enumerate(grads):
        #     self.grads[i][...] = grad

        self.dh = dh
        return dxs