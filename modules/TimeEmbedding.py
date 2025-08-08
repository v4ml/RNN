#import numpy as np
from common.np import *

class TimeEmbedding2:
    def __init__(self, embed_W):
        self.params = [embed_W] # V,D
        self.grads = [np.zeros_like(embed_W)]
        
    def forward(self, xs):
        embed_W = self.params[0]
        V,D = embed_W.shape
        N,T = xs.shape
        returnW = np.zeros((N,T,D)).astype('f')
        
        returnW = embed_W[xs]

        # for t in range(T):
        #     returnW[:, t] =  embed_W[xs[:, t]]

        self.cache = xs

        return returnW
    
    
    def backward(self, dxs):
        N,T,D = dxs.shape
        embed_W = self.params[0]
        V,D = embed_W.shape
        xs = self.cache
        dembed_W = np.zeros((V,D), dtype='f')

        np.add.at(dembed_W, xs, dxs)
        # for n in range(N):
        #     for t in range(T):
        #         dembed_W[xs[n,t]][...] += dxs[n,t]
        
        #dembed_W[xs[:,:]] += dxs[:,:]
               
        # for n in range(N):
        #     for t in range(T):
        #         dembed_W[,:] += dxs[n,t,:]
        self.grads[0][...] = dembed_W
        return dxs

# class TimeEmbedding2:
#     def __init(self, embed_W):
#         self.params 
#         self.grads

#     def forward(self, xs)