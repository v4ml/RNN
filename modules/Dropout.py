from common.np import *
 
class dropout2:
    def __init__(self, ratio=0.25):
        self.params = []
        self.grads = []
        #self.grads = [np.zeros(10), np.zeros(5), np.zeros(100)]
        self.ratio = ratio
        self.cache = []

    def forward(self, xs):
        N,T,A = xs.shape
        self.cache = [self.NAMatrix(N,T,A)]
        xs *= self.cache[0]
        return xs

    def backward(self, dxs):
        N,T,A = dxs.shape
        #self.grads[0] = dxs*self.params[0]
        dxs *= self.cache[0]
        return dxs

    def NAMatrix(self, N, T, A):
        num_zeros = int(A//(1/self.ratio))
        num_ones = A-num_zeros

        base_row = np.array([0]*num_zeros + [1]*num_ones, dtype=int)
        all_time_rows = []
        all_rows = []
        for _ in range(N):
            for _ in range(T):
                shuffled_row = base_row.copy()
                np.random.shuffle(shuffled_row)
                all_time_rows.append(shuffled_row)
            all_rows.append(all_time_rows)
            all_time_rows = []

        matrix = np.array(all_rows)

        return matrix

class dropout:
    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask        