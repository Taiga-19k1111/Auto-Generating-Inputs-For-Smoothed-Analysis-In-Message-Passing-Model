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
            self.add_link('l{}'.format(i), L.Linear(channels[i], channels[i+1], initial_bias=bias[i]))

    def z(self, batch):
        return self.xp.random.randn(batch, self.channels[0]).astype('f')

    def __call__(self, x):
        for i in range(self.n_layers):
            x = self['l{}'.format(i)](x)
            if i + 1 == self.n_layers:
                x = F.relu(x)
                # x = F.sigmoid(x)
                # continue
            else:
                x = F.relu(x)
        return x

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
    tmp = p.data
    total = 0
    for i in range(n*n):
        if post[i] == []:
            tmp[i] = xp.nan
    pi = xp.zeros(n*n)
    exp_tmp = xp.exp(tmp)
    exp_sum = xp.nansum(exp_tmp)
    for i in range(n*n):
        if xp.isnan(exp_tmp[i]):
            pi[i] = 0
        else:
            pi[i] = exp_tmp[i]/exp_sum
    ind = int(xp.random.choice(n*n,1,p=pi))
    send = ind//n
    receive = ind%n
    post[ind].pop(0)
    a = xp.zeros(n*n)
    a[ind] = 1
    lp = F.sum(a * F.log(pi + EPS) + (1 - a) * F.log(1 - pi + EPS))
    a_cpu = chainer.cuda.to_cpu(a)
    return a, [send,receive,G,post], lp
    
def calc_lp(n, p, a, xp):
    EPS = 1e-6
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
    net = MLP(channels, bias)

    if conf['gpu'] != -1:
        chainer.cuda.get_device_from_id(conf['gpu']).use()
        net.to_gpu()
    
    if conf['opt'] == 'SGD':
        opt = chainer.optimizers.SGD(lr=conf['lr'])
    elif conf['opt'] == 'Adam':
        opt = chainer.optimizers.Adam(alpha=conf['lr'])
    opt.setup(net)

    # Mnet Training
    Mchannels = [n*n*2+n, 100, 500, n*n]
    # Mchannels = [10, 100, 500, n*n]
    Mnet = MLP(Mchannels, bias)
    if conf['gpu'] != -1:
        chainer.cuda.get_device_from_id(conf['gpu']).use()
        Mnet.to_gpu()
    
    if conf['opt'] == 'SGD':
        Mopt = chainer.optimizers.SGD(lr=conf['lr'])
    elif conf['opt'] == 'Adam':
        Mopt = chainer.optimizers.Adam(alpha=conf['lr'])
    Mopt.setup(Mnet)

    epoch = conf['epoch']
    p = conf['erp']
    m = conf['message']
    step = conf['step']

    pool_size = 10
    start_training = 20
    r_bests = []
    inputs_bests = []
    z_bests = []

    if no_replay:
        pool_size = 1
        start_training = 1e9

    iteration = 0
    from_restart = 0

    max_r = -1

    memo_x = []
    memo_y = []
    for ep in range(1,epoch+1):
        iteration += 1
        step = 0
        r = 0
        x = 0
        G = gen_worstcase(n)
        G,post,inputs_li,lp = gen_initial_graph_state(G, n, x, m, Mnet.xp)
        memory = Memory()
        check = True
        # target_Mnet = Mnet.copy('copy')
        while check:
            # mx = target_Mnet(Mnet.xp.array([G]).astype('f'))[0]
            mx = Mnet(Mnet.xp.array([G]).astype('f'))[0]
            # z = Mnet.z(1)
            # x = Mnet(z)[0]
            inputs_li, inputs, lp = decide_message(n,mx,G,post,Mnet.xp)
            post, next_G, _ = calc_reward(n, inputs, solver, tmpdir, form)
            memory.add(lp)
            step += 1

            check = False
            for po in post:
                if po != []:
                    check = True
                    break
         
        for i in range(len(memory)):
            loss = memory.buffer[i]*(-step)
            Mnet.cleargrads()
            loss.backward()
            Mopt.update()

        memo_x.append(ep)
        memo_y.append(step)
        # output_graph(os.path.join(savedir, 'output_{}.txt'.format(ep)), n, edges, 0)
        plt.clf()
        plt.plot(memo_x, memo_y)
        plt.savefig(os.path.join(savedir, 'graph.png'))

    G = gen_worstcase(n)
    G,post,inputs_li,lp = gen_initial_graph_state(G, n, x, m, Mnet.xp)
    check = True
    step = 0
    while check:
        step += 1
        mx = Mnet(Mnet.xp.array([G]).astype('f'))[0]
        output_distribution_graph(os.path.join(savedir, 'distribution_{}.txt'.format(step)), n, mx.data)
        inputs_li, inputs, lp = decide_message(n,mx,G,post,Mnet.xp)
        post, G, _ = calc_reward(n, inputs, solver, tmpdir, form)

        check = False
        for po in post:
            if po != []:
                check = True
                break

        # if max_r < step:
        #     max_r = step
        # else:
        #     from_restart += 1
        
        # if from_restart > conf['restart']:
        #     Mnet = MLP(Mchannels, bias)
        #     if conf['gpu'] != -1:
        #         chainer.cuda.get_device_from_id(conf['gpu']).use()
        #         Mnet.to_gpu()
            
        #     if conf['opt'] == 'SGD':
        #         Mopt = chainer.optimizers.SGD(lr=conf['lr'])
        #     elif conf['opt'] == 'Adam':
        #         Mopt = chainer.optimizers.Adam(alpha=conf['lr'])
        #     Mopt.setup(Mnet)
        #     from_restart = 0
        #     max_r = -1

        # G = gen_worstcase(n)
        # G,post,inputs_li,lp = gen_initial_graph_state(G, n, x, m, net.xp)
        # G2,_ = gen_random_graph(n,p,m)
        # message =  np.random.randint(0,m,n*n)
        # for i in range(n*n,n*n*2):
        #     if G[i-n*n] == -1:
        #         G[i] = -1
        #     else:
        #         G[i] = message[i-n*n]
        #         if post[i-n*n] == []:
        #             post[i-n*n].append(message[i-n*n])
        #         else:
        #             post[i-n*n][0] = message[i-n*n]
        # for i in range(n):
        #     G[-i-1] = G2[-i-1]

        # lp = Mnet.xp.zeros(step)
        # reward = Mnet.xp.zeros(step)
        # zero = 0
        # for s in range(step):
        #     x = Mnet(Mnet.xp.array([G]).astype('f'))[0]
        #     inputs_li, inputs, tmp_lp = decide_message(n,x,G,post,Mnet.xp)
        #     lp[s] = tmp_lp.data
        #     post, G, tmp_r = calc_reward(n, inputs, solver, tmpdir, form)
        #     reward[s] = tmp_r
            # if tmp_r == 0:
            #     zero += 1

            # check = False
            # for po in post:
            #     if po != []:
            #         check = True
            #         break

        # if no_replay:
        #     loss = -r * F.sum(lp)
        #     print(loss)
            
        #     Mnet.cleargrads()
        #     loss.backward()
        #     Mopt.update()

        # f = False
        # for ib in inputs_bests:
        #     if (ib == inputs_li).all():
        #         f = True
        # if not f:
        #     r_bests.append(r)
        #     inputs_bests.append(inputs_li)
        
        # while len(r_bests) > pool_size:
        #     mi = 0
        #     for j in range(len(r_bests)):
        #         if r_bests[j] < r_bests[mi]:
        #             mi = j
        #     r_bests.pop(mi)
        #     edges_bests.pop(mi)
        #     z_bests.pop(mi)

        # if from_restart >= start_training:
        #     ind = np.random.randint(len(r_bests))
        #     x = net(z_bests[ind])[0]
        #     lp = calc_lp(n, x, edges_bests[ind], net.xp, form)

        #     loss = - r_bests[ind] * lp

        #     net.cleargrads()
        #     loss.backward()
        #     opt.update()

if __name__ == '__main__':
    train()