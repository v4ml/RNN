import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BetterRnnlm import BetterRnnlm


class Encoder:
    def __init__(self, xs):
        self.rnnlm = BetterRnnlm(7, 20, 20)


    def forward(self, xs):
        #N, T, H = xs.shape
        hs = self.rnnlm.forward(xs)
        return hs
    
    def backward(self):
        pass

class Decoder:
    def __init__(self):
        pass

    def forward(self, hs):
        pass
    
    def backward(self):
        pass
    


