from encoder import Encoder
from encoder import Decoder

from common.np import *

class Seq2seq:
    def __init__(self, encoder, decoder, batch_size, hidden_size, time_size):
        self.encoder = encoder
        self.decoder = decoder

        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.time_size = time_size

        
    def forward(self, xs, ts):
        tts = np.zeros_like(ts, dtype=np.int32) #np.zeros((N,Tt), dtype=t_train.dtype)
        tts[:, 4] = 5
        #print("t_train.shape :  ", ts.shape)
        #print("tt_train.shape :  ", tts.shape)
        tts[:, :4] = np.array(ts[:, 1:5], dtype=tts.dtype)

        N, T = xs.shape
        batch = N//self.batch_size

        for i in range(batch):
            hs = self.encoder.forward(xs[self.batch_size*i:self.batch_size*(i+1), :])#, ts[self.batch_size*i:self.batch_size*(i+1), :])    
            loss = self.decoder.forward(ts[self.batch_size*i:self.batch_size*(i+1), :], tts[self.batch_size*i:self.batch_size*(i+1), :], hs[:, -1])

        return loss
        
    def backward(self):
        dout, dh = self.decoder.backward()
        dh = dh.reshape(self.batch_size,1, self.hidden_size)
        dh = np.repeat(dh, 7, axis=1)
        dout = self.encoder.backward(dh)    

        return dout
    
    def generate(self, question, start_id, length):
        N, _ = question.shape
        batch = N//self.batch_size
        for i in range(batch):
            hs = self.encoder.forward(question[i*20:(i+1)*20, :])
            ts = self.decoder.generate(start_id, length, hs[:, -1])
        return ts