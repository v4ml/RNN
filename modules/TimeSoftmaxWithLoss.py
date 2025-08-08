import numpy as np
from TimeEmbedding import TimeEmbedding2

class TimeSoftmaxWithLoss3:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None


    def forward(self, xs, ts):
        N,T,V = xs.shape
        self.params = [xs, ts]
        loss = np.zeros((N,T), dtype='f')
        
        down = np.zeros((N,T), dtype='f')
        
        ys = np.zeros((N,T,V), dtype='f')
        ys = np.exp( xs )
        down = np.sum( ys, axis=2 )
        
        ys = ys / down[:,:, np.newaxis]

        loss = -np.log(ys[np.arange(N)[:, None], range(T), ts])
        
        self.cache = ts, xs, ys, (N, T, V)
        loss = np.sum(loss)/(N*T)
        return loss


    def backward(self, dout=1):
        xs, ts = self.params
        ts, xs, ys, (N,T,V) = self.cache # ys바꿔서 1회용이면 괜찮은지 테스트

        ys = ys.reshape(N*T, -1)
        ts = ts.reshape(N*T)
        ys[np.arange(N*T), ts] -= 1
        ys.reshape(N,T,-1)
        return ys/(N*T)*dout



class TimeSoftmaxWithLoss2:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N,T,V = xs.shape
        self.params = [xs, ts]
        loss = np.zeros((N,T), dtype='f')
        #exp = np.exp(xs+1e-7)
        
        #down = np.exp(np.sum(xs, axis=2))
        #out = exp / np.sum(exp+1e-7)
        down = np.zeros((N,T), dtype='f')
        temp = np.zeros((N,T,V), dtype='f')
        #xs = xs - np.max(xs, axis=2, keepdims=True)
        #xs = xs - np.max(xs, axis=2, keepdims=True)
        temp = np.exp( xs )
        ys = np.zeros((N,T,V), dtype='f')

        down = np.sum( temp, axis=2 )
        for v in range(V):
            ys[:,:,v] = temp[:,:,v] / down[:,:]

        for n in range(N):
            for t in range(T):
                #loss[n,t] = np.exp(xs[n, t, ts[n,t]])
                down[n,t] = np.sum(np.exp(xs[n,t,:]))
                # ys[n,t] = ys[n, t, ts[n,t]]/((down[n,t])+1e-7)
                loss[n,t] = -np.log(ys[n, t, ts[n,t]])
        
        self.cache = ts, xs, ys, (N, T, V)
        loss = np.sum(np.sum(loss, axis=1), axis=0)
        return loss/(N*T)


    def backward(self, dout=1):
        xs, ts = self.params
        ts, xs, ys, (N,T,V) = self.cache

        one_hot = np.eye(V)[ts]
        temp = np.zeros((N,T,V), dtype='f')
        temp = ys-one_hot
        return (ys-one_hot)/(N*T)
