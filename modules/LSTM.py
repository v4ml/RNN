from common.np import *

class LSTM2:
    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def sigmoid(self, x):
        #print("sigmoid input stats: min", np.min(x), "max", np.max(x))
        #x = np.clip(x, -709, 709)
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        H = h_prev.shape[1]

        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b  # (4h,)

        f = self.sigmoid(A[:, :H])
        g = np.tanh(A[:, H:2*H])
        i = self.sigmoid(A[:, 2*H:3*H])
        o = self.sigmoid(A[:, 3*H:4*H])

        # i = self.sigmoid(A[:, :H])
        # f = self.sigmoid(A[:, H:2*H])
        # o = self.sigmoid(A[:, 2*H:3*H])
        # g = np.tanh(A[:, 3*H:4*H])

        c_next = f*c_prev + g*i
        h_next = np.multiply(o, np.tanh(c_next))

        self.cache = [x,h_prev,c_prev,c_next,f,g,i,o]

        return h_next, c_next
    

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x,h,c_prev,c_next,f,g,i,o = self.cache
        tanh_c_next = np.tanh(c_next)
        ds = dc_next+(dh_next*o)*(1-tanh_c_next**2)

        H = Wh.shape[0]

        #slice = np.zeros((10, H*4), dtype='f')
        #dc_prev = np.zeros((10,80), dtype='f')
        #dh_prev = np.zeros((10,80), dtype='f')
        
        # do
        do = dh_next*tanh_c_next*o*(1-o)
        # di
        di = ds*g*i*(1-i)
        # dg
        dg = ds*i*(1-g**2)
        # df
        df = ds*c_prev*f*(1-f)
        dA = np.hstack((di, df, do, dg))

        dx = np.matmul(dA, Wx.T)
        dWx = np.matmul(x.T, dA)
        db = np.sum(dA, axis=0)
        dc_prev = ds*f
        dh_prev = np.matmul(dA, Wh.T)
        dWh = np.matmul(h.T, dA)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev, dc_prev



class LSTM:
    def __init__(self, Wx, Wh, b):
        '''
        Parameters
        ----------
        Wx: 입력 x에 대한 가중치 매개변수(4개분의 가중치가 담겨 있음)
        Wh: 은닉 상태 h에 대한 가장추 매개변수(4개분의 가중치가 담겨 있음)
        b: 편향（4개분의 편향이 담겨 있음）  
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        N, H = h_prev.shape
        
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b
        
        f = A[:, :H]
        g = A[:, H:2*H]
        i = A[:, 2*H:3*H]
        o = A[:, 3*H:]
        
        f = self.sigmoid(f)
        g = np.tanh(g)
        i = self.sigmoid(i)
        o = self.sigmoid(o)
        
        c_next = f * c_prev + g * i  # Ct
        h_next = o * np.tanh(c_next)
        
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next
    
    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache
        
        tanh_c_next = np.tanh(c_next)
        
        ds = dc_next + (dh_next * o) * (1 - tanh_c_next ** 2)
        
        dc_prev = ds * f
        
        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i
        
        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g ** 2)
        
        dA = np.hstack((df, dg, di, do))
        
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)
        
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)
        
        return dx, dh_prev, dc_prev