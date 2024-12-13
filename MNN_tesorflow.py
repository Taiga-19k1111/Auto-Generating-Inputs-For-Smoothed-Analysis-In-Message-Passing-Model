import os
import time

import matplotlib
matplotlib.use('Agg')

import numpy as np
import cupy as cp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.keras.losses import Huber

import matplotlib.pyplot as plt
import yaml

from util import calc_reward, makedir, output_graph, output_distribution_graph, gen_random_graph, load_conf
from gen_SSSP_worstcase import gen_worstcase

class QNetwork:
    def __init__(self, state_size, action_size):
        self.model = Sequential()
        self.model.add(Dense(16, activation='relu', input_shape=state_size))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(action_size, activation='linear'))

        self.model.compile(loss=Huber(), optimizer=Adam(learning_rate=0.001))

class Memory():
    def __init__(self, memory_size=100000):
        self.buffer = deque(maxlen=memory_size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[int(i)] for i in idx]
    
    def __len__(self):
        return len(self.buffer)
    
def gen_initial_graph_state(g, n):
    post = [[] for _ in range(n*n)]
    G = cp.ones([(n*2)+1,n], dtype=float)*-1

    for i in range(n):
        for j in range(n):
            G[i][j] = g[i][j]

    start_node = 0
    for i in range(n):
        if G[start_node][i] != -1:
            G[start_node+n][i] = G[start_node][i]
            post[start_node*n+i].append(int(G[start_node][i]))
        G[n*2][i] = 10**8
    G[n*2][start_node] = 0

    return cp.asnumpy(G.ravel()),post

def decide_message(n, p, G, post):
    memo = []
    for i in range(n*n):
        if post[i] != []:
            memo.append([p[i],i])
    memo = sorted(memo, reverse=True)
    ind = memo[0][1]
    send = ind//n
    receive = ind%n
    post[ind].pop(0)

    return [send,receive,G,post]

def train():
    conf = load_conf()

    no_replay = conf['noreplay']
    
    solver = conf['solver']
    n = conf['n']
    eps = conf['eps']

    form = conf['form']

    savedir = conf['dirname']
    makedir(savedir)
    tmpdir = os.path.join(savedir, 'tmp')
    makedir(tmpdir)

    E_START = 1.0
    E_STOP = 0
    E_DECAY_RATE = 0.0001
    GAMMA = 0.99

    np.random.seed(conf['seed'])
    cp.random.seed(conf['seed'])

    logfile = os.path.join(savedir, 'log')

    ave = 0
    aves = []
    ma = 0
    global_ma = 0

    G = gen_worstcase(n)
    G,post = gen_initial_graph_state(G, n)
    main_qn = QNetwork((n*n,), n*n)
    target_qn = QNetwork((n*n,), n*n)

    epoch = conf['epoch']
    p = conf['erp']
    m = conf['message']
    step = conf['step']

    pool_size = 10
    start_training = 5
    r_bests = []
    inputs_bests = []
    z_bests = []

    if no_replay:
        pool_size = 1
        start_training = 1e9

    iteration = 0
    from_restart = 0

    memory = Memory()

    memo_x = []
    memo_y = []
    memo_ave = []
    ave = 0
    total_step = 0
    total_max = 0
    for ep in range(1,epoch+1):
        cmnn = 0
        step = 0
        x = 0
        G = gen_worstcase(n)
        G,post = gen_initial_graph_state(G, n)
        edges = []
        check = True

        target_qn.model.set_weights(main_qn.model.get_weights())

        while check:
            total_step += 1
            step += 1
            epsilon = E_STOP+(E_START-E_STOP)*np.exp(-E_DECAY_RATE*total_step)

            if epsilon > np.random.uniform(0,1,1):
                mx = np.random.uniform(0,1,n*n)
                ep_check = 0
            else:
                cmnn += 1
                mx = main_qn.model.predict(np.array([G[n*n:n*n*2]]))[0]
                ep_check = 1

            # if ep%100 == 0:
            #     output_distribution_graph(os.path.join(savedir, 'distribution_{}_{}_{}.txt'.format(ep,step,ep_check)), n, mx)

            inputs = decide_message(n,mx,G,post)
            post, next_G, r = calc_reward(n, inputs, solver, tmpdir, form)
            edges.append(inputs[0:2])

            check = False
            for po in post:
                if po != []:
                    check = True
                    break

            # if check:
            #     reward = 0
            # else:
            #     reward = step
            reward = r

            if step > start_training:
                memory.add([G,inputs[0]*n+inputs[1],reward,next_G])

            G = next_G
            if len(memory) >= pool_size and total_step >= 1000:
                inputs = np.zeros([pool_size,n*n])
                targets = np.zeros([pool_size,n*n])

                minibatch = memory.sample(pool_size)
                for i, [G_b,inputs_b,reward_b,next_G_b] in enumerate(minibatch):
                    inputs[i] = G_b[n*n:n*n*2]
                    next_initial_post = (next_G_b[n*n:n*n*2] != -1)
                    if next_initial_post.any():
                        next_q = target_qn.model.predict(np.array([next_G_b[n*n:n*n*2]]))[0]
                        max_q = np.amax(next_q*np.where(next_initial_post,1,0))
                        target = reward_b + GAMMA*max_q
                    else:
                        target =reward_b
                    targets[i] = main_qn.model.predict(np.array([inputs[i]]))[0]
                    targets[i][inputs_b] = target
                main_qn.model.fit(inputs, targets, epochs=1, verbose=0)

        memo_x.append(ep)
        memo_y.append(step)
        ave = ((ep-1)*ave+step)/ep
        memo_ave.append(ave)
        plt.clf()
        plt.plot(memo_x, memo_y)
        plt.savefig(os.path.join(savedir, 'message_num.png'))
        plt.clf()
        plt.plot(memo_x, memo_ave)
        plt.savefig(os.path.join(savedir, 'ave_message_num.png'))

        if step > total_max:
            total_max = step
            output_graph(os.path.join(savedir, 'output_{}.txt'.format(total_max)), n, edges, 0)


        print(epsilon, np.mean(memo_y), cmnn)

if __name__ == '__main__':
    train()