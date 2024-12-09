import os
import time

import matplotlib
matplotlib.use('Agg')

from collections import deque
import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L
import matplotlib.pyplot as plt
import yaml

from util import calc_reward, makedir, output_graph, output_distribution_graph, gen_random_graph, load_conf
from gen_SSSP_worstcase import gen_worstcase

class MLP(chainer.Chain):
    def __init__(self, channels, bias_final):
        super(MLP, self).__init__()
        self.n_layers = len(channels) - 1
        self.channels = channels
        bias = [0 for i in range(self.n_layers)]
        bias[-1] = bias_final
        for i in range(self.n_layers):
            self.add_link('l{}'.format(i), L.Linear(channels[i], channels[i+1]))
            # self.add_link('bn{}'.format(i), L.BatchNormalization(channels[i+1]))

    def z(self, batch):
        return self.xp.random.randn(batch, self.channels[0]).astype('f')

    def __call__(self, x):
        for i in range(self.n_layers):
            x = self['l{}'.format(i)](x)
            # x = self['bn{}'.format(i)](x)
            if i + 1 == self.n_layers:
                # x = F.relu(x)
                continue
            else:
                x = F.relu(x)
        return x
    
    def set_parameter(self, i, w, b=[]):
        self['l{}'.format(i)].W = w
        if len(b) != 0:
            self['l{}'.format(i)].b = b

    def get_parameter(self, i):
        return self['l{}'.format(i)].W, self['l{}'.format(i)].b

class Memory():
    def __init__(self, memory_size=100000):
        self.buffer = deque(maxlen=memory_size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[i] for i in idx]
    
    def __len__(self):
        return len(self.buffer)
    
def gen_edges(n, p, xp):
    EPS = 1e-6
    a = xp.random.binomial(1, p.data, n * (n-1) // 2)
    lp = F.sum(a * F.log(p + EPS) + (1 - a) * F.log(1 - p + EPS))
    a_cpu = chainer.cuda.to_cpu(a)
    edges = np.array(np.tril_indices(n, -1)).T[np.where(a_cpu == 1)]
    return a, edges, lp
    
def gen_initial_graph_state(g, n, p, m, xp):
    EPS = 1e-6

    # a = xp.random.binomial(1, p.data[:n*(n-1)//2], n*(n-1)//2)
    post = [[] for _ in range(n*n)]
    G = xp.ones([(n*2)+1,n], dtype=float)*-1
    # count = 0
    # for i in range(n-1):
    #     for j in range(i+1,n):
    #         G[i+n][j] = p.data[(n*(n-1)//2)+count]%m
    #         if a[count] == 1:
    #             G[i][j] = p.data[(n*(n-1)//2)+count]
    #             G[j][i] = G[i][j]
    for i in range(n):
        for j in range(n):
            G[i][j] = g[i][j]
    # tmp = p.data[-n:]
    # total = sum(tmp)
    # rnd = xp.random.uniform(0,total)
    # cum = 0
    # for i in range(n):
    #     cum += tmp[i]
    #     if rnd < cum:
    #         start_node = i
    #         break
    start_node = 0
    for i in range(n):
        if G[start_node][i] != -1:
            G[start_node+n][i] = G[start_node][i]
            post[start_node*n+i].append(int(G[start_node][i]))
        G[n*2][i] = 10**8
    G[n*2][start_node] = 0

    # a = xp.concatenate([a,xp.asarray(G[n*2])])
    # lp = F.sum(a * F.log(p + EPS) + (1 - a) * F.log(1 - p + EPS))
    a = 0
    lp = 0

    return xp.array(G.ravel()),post,a,lp

def decide_message(n, p, G, post, xp):
    EPS = 1e-6
    # send,receive = [-1,-1]
    # total = 0
    # for i in range(n):
    #     for j in range(n):
    #         ind = i*n+j
    #         if post[ind] == []:
    #             tmp[ind] = 0
    #         else:
    #             tmp[ind] += EPS
    #         total += tmp[ind]
    #         # if mx < tmp[ind]:
    #         #     mx = tmp[ind]
    #         #     send = i
    #         #     receive = j
    # rnd = xp.random.uniform(0,total)
    # cum = 0
    # for i in range(n):
    #     for j in range(n):
    #         ind = i*n+j
    #         cum += tmp[ind]
    #         if rnd < cum:
    #             send = i
    #             receive = j
    #             break
    #     if send != -1:
    #         break
    # ind = int(xp.argmax(p))
    # if post[ind] == []:
    #     print(xp.array(post))

    # ind = int(xp.argmax(p))
    memo = []
    for i in range(n*n):
        if post[i] != []:
            memo.append([p[i],i])
    # memo = sorted(memo, reverse=True)
    memo = sorted(memo)
    ind = memo[0][1]
    send = ind//n
    receive = ind%n
    post[ind].pop(0)
    a = xp.zeros((n,n))
    a[send][receive] = 1
    a = a.ravel()
    # lp = F.sum(a * F.log(p + EPS) + (1 - a) * F.log(1 - p + EPS))
    lp = 0
    a_cpu = chainer.cuda.to_cpu(a)
    return a, [send,receive,G,post], lp
    
def calc_lp(n, p, a, xp, f):
    EPS = 1e-6
    if f == 0:
        lp = F.sum(a * F.log(p + EPS) + (1 - a) * F.log(1 - p + EPS))
    elif f == 1:
        lp = F.sum(a * F.log(p + EPS) + (1 - a) * F.log(1 - p + EPS))
    return lp

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
    E_DECAY_RATE = 0.00001
    GAMMA = 0.99

    np.random.seed(conf['seed'])
    if chainer.cuda.available:
        chainer.cuda.cupy.random.seed(conf['seed'])

    logfile = os.path.join(savedir, 'log')

    ave = 0
    aves = []
    ma = 0
    global_ma = 0

    channels = [10, 100, 500, (n*(n-1)//2)+n]
    if 'channels' in conf:
        channels = conf['channels']
        channels.append((n*(n-1)//2)+n)

    bias = - np.log(1.0 / conf['p']  - 1)
    # net = MLP(channels, bias)

    # if conf['gpu'] != -1:
    #     chainer.cuda.get_device_from_id(conf['gpu']).use()
    #     net.to_gpu()
    
    # if conf['opt'] == 'SGD':
    #     opt = chainer.optimizers.SGD(lr=conf['lr'])
    # elif conf['opt'] == 'Adam':
    #     opt = chainer.optimizers.Adam(alpha=conf['lr'])
    # opt.setup(net)

    # Mnet Training
    # Mchannels = [n*n*2+n, n*10, n*n/2, n*n]
    Mchannels = [n*n+n, 100, 500, n*n]
    Mnet = MLP(Mchannels, bias)
    target_Mnet = MLP(Mchannels, bias)
    if conf['gpu'] != -1:
        chainer.cuda.get_device_from_id(conf['gpu']).use()
        Mnet.to_gpu()
        target_Mnet.to_gpu()
    
    if conf['opt'] == 'SGD':
        Mopt = chainer.optimizers.SGD(lr=conf['lr'])
        target_Mopt = chainer.optimizers.SGD(lr=conf['lr'])
    elif conf['opt'] == 'Adam':
        Mopt = chainer.optimizers.Adam(alpha=conf['lr'])
        target_Mopt = chainer.optimizers.Adam(alpha=conf['lr'])
    Mopt.setup(Mnet)
    target_Mopt.setup(target_Mnet)

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
    total_step = 0
    total_max = 0
    for ep in range(1,epoch+1):
        # iteration += 1
        # from_restart += 1
        # if ep%10 == 0:
        # lp = Mnet.xp.zeros(0)
        cmnn = 0
        step = 0
        x = 0
        G = gen_worstcase(n)
        G,post,inputs_li,lp = gen_initial_graph_state(G, n, x, m, Mnet.xp)
        edges = []
        check = True

        target_Mnet = Mnet.copy('copy')
        while check:
            total_step += 1
            step += 1
            epsilon = E_STOP+(E_START-E_STOP)*np.exp(-E_DECAY_RATE*total_step)

            if epsilon > Mnet.xp.random.uniform(0,1,1):
                rnd = Mnet.xp.random.uniform(0,1,n*n)
                mx = Mnet.xp.where(G[n*n:n*n*2] > -1,1,0)*rnd
                ep_check = 0
            else:
                cmnn += 1
                mx = Mnet.xp.where(G[n*n:n*n*2] > -1,1,0)*Mnet(Mnet.xp.array([G[n*n:]]).astype('f'))[0].data
                # if post[int(Mnet.xp.argmax(mx))] == []:
                #     loss = F.mean_squared_error(Mnet.xp.amax(mx).astype('f'),Mnet.xp.array(-100).astype('f'))
                #     Mnet.cleargrads()
                #     loss.backward()
                #     Mopt.update()
                #     break
                ep_check = 1
            if ep%100 == 0:
                output_distribution_graph(os.path.join(savedir, 'distribution_{}_{}_{}.txt'.format(ep,step,ep_check)), n, mx)
            inputs_li, inputs, lp = decide_message(n,mx,G,post,Mnet.xp)
            post, next_G, r = calc_reward(n, inputs, solver, tmpdir, form)
            edges.append(inputs[0:2])

            check = False
            for po in post:
                if po != []:
                    check = True
                    break

            if check:
                reward = 0
            else:
                reward = 1-1/step

            if step > start_training:
                memory.add([G,inputs[0]*n+inputs[1],reward,next_G])

            G = next_G
            if len(memory) >= pool_size:
                # inputs = Mnet.xp.zeros([pool_size,channels[0]])
                # targets = Mnet.xp.zeros([pool_size,channels[-1]])

                minibatch = memory.sample(pool_size)
                for i, [G_b,inputs_b,reward_b,next_G_b] in enumerate(minibatch):
                    now_initial_post = G_b[n*n:n*n*2]
                    outputs = Mnet(Mnet.xp.array([G_b[n*n:]]).astype('f'))[0].data
                    targets = outputs.copy()
                    next_outputs = target_Mnet(Mnet.xp.array([next_G_b[n*n:]]).astype('f'))[0].data
                    # targets[Mnet.xp.where(now_initial_post == -1)[0]] = -1000
                    next_initial_post = (next_G_b[n*n:n*n*2] != -1)
                    if next_initial_post.any():
                        max_q = Mnet.xp.amax(next_outputs[Mnet.xp.where(next_initial_post)[0]])
                        target = reward_b + GAMMA*max_q
                    else:
                        target = Mnet.xp.array(reward_b).astype('f')
                    targets[inputs_b] = target
                    loss = F.mean_squared_error(targets,outputs)
                    Mnet.cleargrads()
                    loss.backward()
                    Mopt.update()

        memo_x.append(ep)
        memo_y.append(step)
        output_graph(os.path.join(savedir, 'output_{}.txt'.format(ep)), n, edges, 0)
        plt.clf()
        plt.plot(memo_x, memo_y)
        plt.savefig(os.path.join(savedir, 'graph.png'))

        print(epsilon, np.mean(memo_y), cmnn)
 
    # G = gen_worstcase(n)
    # G,post,inputs_li,lp = gen_initial_graph_state(G, n, x, m, Mnet.xp)
    # check = True
    # step = 0
    # while check:
    #     step += 1
    #     mx = Mnet.xp.where(G[n*n:n*n*2] > -1,1,0)*Mnet(Mnet.xp.array([G[n*n:]]).astype('f'))[0].data
    #     # if post[int(Mnet.xp.argmax(mx))] == []:
    #     #     rnd = Mnet.xp.random.uniform(0,1,n*n)
    #     #     mx = Mnet.xp.where(G[n*n:n*n*2] > -1,1,0)*rnd
        
    #     output_distribution_graph(os.path.join(savedir, 'distribution_{}.txt'.format(step)), n, mx)
    #     inputs_li, inputs, lp = decide_message(n,mx,G,post,Mnet.xp)
    #     post, G, _ = calc_reward(n, inputs, solver, tmpdir, form)

    #     check = False
    #     for po in post:
    #         if po != []:
    #             check = True
    #             break

if __name__ == '__main__':
    train()