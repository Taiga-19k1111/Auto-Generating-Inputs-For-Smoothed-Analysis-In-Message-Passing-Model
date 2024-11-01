import os
import time

import matplotlib
matplotlib.use('Agg')

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
                x = F.sigmoid(x)
            else:
                x = F.relu(x)
        return x

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
    mx = -1
    send,receive = [-1,-1]
    for i in range(n):
        for j in range(n):
            ind = i*n+j
            if i == j:
                continue
            if G[ind] == -1:
                continue
            if post[ind] == []:
                continue

            if mx < tmp[ind]:
                mx = tmp[ind]
                send = i
                receive = j
    post[(send*n)+receive].pop(0)
    a = xp.zeros((n,n))
    a[send][receive] = 1
    a = a.ravel()
    lp = F.sum(a * F.log(p + EPS) + (1 - a) * F.log(1 - p + EPS))
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

    np.random.seed(conf['seed'])

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
    for ep in range(epoch):
        G, post = gen_random_graph(n,p,m)
        x = Mnet(Mnet.xp.array([G]).astype('f'))[0]
        inputs_li, inputs, lp = decide_message(n,x,G,post,Mnet.xp)
        _,_,r = calc_reward(n, inputs, solver, tmpdir, form)
        loss = - r * lp

        Mnet.cleargrads()
        loss.backward()
        Mopt.update()

    r = 0
    x = 0
    G = gen_worstcase(n)
    G,post,inputs_li,lp = gen_initial_graph_state(G, n, x, m, net.xp)
    check = True
    while check:
        mx = Mnet(Mnet.xp.array([G]).astype('f'))[0]
        inputs_li, inputs, lp = decide_message(n,mx,G,post,Mnet.xp)
        post,G,_ = calc_reward(n, inputs, solver, tmpdir, form)
        r += 1
        print(r,inputs[0],inputs[1])

        check = False
        for p in post:
            if p != []:
                check = True
                break

if __name__ == '__main__':
    train()