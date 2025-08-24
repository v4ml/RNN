import numpy as np

class TimeAffine:
    def __init__(self, affine_W, affine_b):
        self.params = [affine_W, affine_b]
        self.grads = [np.zeros_like(affine_W), np.zeros_like(affine_b)]

    def forward(self, hs):
        N,T,H = hs.shape
        affine_W, affine_b = self.params
        out = np.einsum('nth,nthv->ntv', hs, affine_W)+affine_b.reshape(N, T, 1)
        self.cache = hs, affine_W
        return out
    
    def backward(self, dout):
        hs, affine_W = self.cache
        #daffine_W = np.matmul(h.T, dout)
        daffine_W = np.einsum('nth,ntv->nthv', hs, dout)
        #dh = np.matmul(dout, affine_W.T)
        dh = np.einsum('ntv,nthv->nth', dout, affine_W)
        daffine_b = np.sum(dout, axis=2)
        self.grads[0][...] = daffine_W
        self.grads[1][...] = daffine_b
        return dh

class TimeAffine2:
    def __init__(self, affine_W, affine_b):
        self.params = [affine_W, affine_b]
        self.grads = [np.zeros_like(affine_W), np.zeros_like(affine_b)]

    def forward(self, hs):
        N,T,H = hs.shape
        affine_W, affine_b = self.params
        hs = hs.reshape(N*T, -1)
        out = np.matmul(hs, affine_W)+affine_b
        
        hs = hs.reshape(N,T,-1)
        out = out.reshape(N,T,-1)

        self.cache = hs, affine_W
        return out
    
    def backward(self, dout):
        hs, affine_W = self.cache
        #N,T,H = hs.shape
        N,T,D = dout.shape
        T = 5
        #daffine_W = np.einsum('nth,ntv->nthv', hs, dout)
        hs = hs.reshape(N*T, -1) # N*TH
        dout = dout.reshape(N*T, -1)  #N*TV
        daffine_W = np.dot(hs.T, dout)

        dh = np.dot(dout, affine_W.T)
        dh = dh.reshape(N,T,-1)
        
        hs = hs.reshape(N,T,-1)
        self.cache = hs, affine_W

        daffine_b = np.sum(dout, axis=0)

        self.grads[0][...] = daffine_W
        self.grads[1][...] = daffine_b
        return dh