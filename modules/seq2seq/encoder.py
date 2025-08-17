import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from BetterRnnlm import BetterRnnlm
from BetterRnnlm import BetterRnnlm2


class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        self.rnnlm = BetterRnnlm2(vocab_size, wordvec_size, hidden_size)
        #self.rnnlm.load_params('./BetternRnnlm.pkl')


    def forward(self, xs, ts):
        #N, T, H = xs.shape
        hs = self.rnnlm.forward(xs, ts)
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
    


