import numpy as np
from Affine import Affine

class TimeAffine:
    def __init__(self, affine_W, affine_b):
        self.params = affine_W, affine_b


    def forward(self, xs):
        N, T, D = xs.shape
        affine_W, affine_b = self.params
        
        if t in range(T):
            Affine(xs[:, t, :])
            
        np.matmul(xs, affine_W) + 

    def backward(self, dout):

