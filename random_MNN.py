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
    
def gen_initial_graph_state(g, n):
    EPS = 1e-6

    post = [[] for _ in range(n*n)]
    G = np.ones([(n*2)+1,n], dtype=float)*-1
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

    return G.ravel(),post

def decide_message(n, G, post):
    EPS = 1e-6
    tmp = np.zeros(n*n)
    send,receive = [-1,-1]
    total = 0
    for i in range(n):
        for j in range(n):
            ind = i*n+j
            if post[ind] == []:
                continue
            tmp[ind] = 1
            total += 1

    rnd = np.random.uniform(0,total)
    cum = 0
    for i in range(n):
        for j in range(n):
            ind = i*n+j
            cum += tmp[ind]
            if rnd < cum:
                send = i
                receive = j
                break
        if send != -1:
            break
    post[(send*n)+receive].pop(0)
    return [send,receive,G,post]
    
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

    epoch = conf['epoch']
    p = conf['erp']
    m = conf['message']

    memo_x = []
    memo_y = []
    for ep in range(0,epoch+1):
        r = 0
        G = gen_worstcase(n)
        G,post = gen_initial_graph_state(G, n)
        edges = []
        check = True
        while check:
            inputs = decide_message(n, G, post)
            post,G,_ = calc_reward(n, inputs, solver, tmpdir, form)
            r += 1
            edges.append(inputs[0:2])

            check = False
            for po in post:
                if po != []:
                    check = True
                    break

        memo_x.append(ep)
        memo_y.append(r)
        output_graph(os.path.join(savedir, 'output_{}.txt'.format(ep)), n, edges, 0)
        plt.clf()
        plt.plot(memo_x, memo_y)
        plt.savefig(os.path.join(savedir, 'graph.png'))

        ave = ave * (1 - conf['eps']) + r * conf['eps']
        aves.append(ave)
        plt.clf()
        plt.plot(memo_x, aves)
        plt.savefig(os.path.join(savedir, 'ave.png'))

if __name__ == '__main__':
    train()