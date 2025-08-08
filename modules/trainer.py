from signal import SIG_DFL
import sys
import time
sys.path.append('..')
#import matplotlib.pyplot as plt
from dataset import ptb
#import numpy as np
from common.np import *
from SimpleRnnlm import SimpleRnnlm

from common.optimizer import SGD
from util import clip_grads
import matplotlib.pyplot as plt
from LSTM_LM import Rnnlm


def remove_duplicate(params, grads):
    '''
    매개변수 배열 중 중복되는 가중치를 하나로 모아
    그 가중치에 대응하는 기울기를 더한다.
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 가중치 공유 시
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 경사를 더함
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 가중치를 전치행렬로 공유하는 경우(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

class trainer2:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.ppl_list = []
        self.current_epoch = 0

    def fit(self, xs, ts, corpus_size, max_epoch, batch_size, time_size, max_grad, eval_interval=20):
        max_iters = len(xs) // (batch_size * time_size)
        offset = 0
        total_loss = 0
        loss_count = 0
        
        start_time = time.time()
        #or epoch in range(max_epoch):
        for iters in range(max_iters):
                idx = np.empty((batch_size, time_size), dtype='i')
                x = [] #np.arange(batch_size, 5)
                t = []
                #temp = np.arange(5)
                
                for batch_idx in range(batch_size):
                    start_idx = ((batch_idx*time_size)+offset)%(corpus_size-1)
                    #idx[batch_idx] = [(i%(corpus_size-1)) for i  in range(start_idx, start_idx+time_size)]
                    idx[batch_idx] = np.array([(i % (corpus_size - 1)) for i in range(start_idx, start_idx + time_size)])

                    #print(idx[batch_idx])
                    #print(xs[idx[batch_idx]])
                    #temp = xs[idx[batch_idx]]
                    x.append(xs[idx[batch_idx]])
                    t.append(ts[idx[batch_idx]])

                offset = offset+time_size
                x = np.array(x)
                t = np.array(t)
                #print(x)
                loss = self.model.forward(x, t)
                #print(model.grads[1].__array_interface__['data'])
                self.model.backward()
                #print(model.grads[1].__array_interface__['data'])
                clip_grads(self.model.grads, max_grad)

                self.optimizer.update(self.model.params, self.model.grads)


                #print('learning : %d' % model.grads[2].__array_interface__['data'][0])
                #print('loss %d' % (loss))
                total_loss += loss
                loss_count += 1

                #print(x)
                #print("=========== iter end===")
                # 퍼플렉서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
        self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x.get(), self.ppl_list, label='train')
        plt.xlabel('반복 (x' + str(20) + ')')
        plt.ylabel('손실')
        plt.show()
            #print("=========== "+str(epoch)+" epoch end===================")


class trainer3:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.ppl_list = []
        self.current_epoch = 0

    def fit(self, xs, ts, corpus_size, max_epoch, batch_size, time_size, max_grad, eval_interval=20):
        max_iters = len(xs) // (batch_size * time_size)
        offset = 0
        total_loss = 0
        loss_count = 0
        
        start_time = time.time()
        #or epoch in range(max_epoch):
        for iters in range(max_iters):
                idx = np.empty((batch_size, time_size), dtype='i')
                x = [] #np.arange(batch_size, 5)
                t = []
                #temp = np.arange(5)
                
                for batch_idx in range(batch_size):
                    start_idx = ((batch_idx*(len(xs)//batch_size))+offset)%(corpus_size-1)
                    #idx[batch_idx] = [(i%(corpus_size-1)) for i  in range(start_idx, start_idx+time_size)]
                    idx[batch_idx] = np.array([(i % (corpus_size - 1)) for i in range(start_idx, start_idx + time_size)])

                    #print(idx[batch_idx])
                    #print(xs[idx[batch_idx]])
                    #temp = xs[idx[batch_idx]]
                    x.append(xs[idx[batch_idx]])
                    t.append(ts[idx[batch_idx]])

                offset = offset+time_size
                x = np.array(x) 
                t = np.array(t)
                #print(x)
                loss = self.model.forward(x, t)
                #print(model.grads[1].__array_interface__['data'])
                self.model.backward()
                #print(model.grads[1].__array_interface__['data'])
                clip_grads(self.model.grads, max_grad)

                self.optimizer.update(self.model.params, self.model.grads)


                #print('learning : %d' % model.grads[2].__array_interface__['data'][0])
                #print('loss %d' % (loss))
                total_loss += loss
                loss_count += 1

                #print(x)
                #print("=========== iter end===")
                # 퍼플렉서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0
        self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x.get(), self.ppl_list, label='train')
        plt.xlabel('반복 (x' + str(20) + ')')
        plt.ylabel('손실')
        plt.show()
            #print("=========== "+str(epoch)+" epoch end===================")


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]  # 배치에서 각 샘플을 읽기 시작하는 위치

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35,
            max_grad=None, eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # 기울기를 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)  # 공유된 가중치를 하나로 모음
                #params, grads = model.params, model.grads
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 퍼플렉서티 평가
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d |  반복 %d / %d | 시간 %d[s] | 퍼플렉서티 %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1
            
    def plot(self, ylim=None):
        x = np.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('퍼플렉서티')
        plt.show()            