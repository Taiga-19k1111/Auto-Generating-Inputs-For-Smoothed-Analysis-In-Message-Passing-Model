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

from util import calc_reward, makedir, output_graph, output_distribution_graph, load_conf

class MLP(chainer.Chain):
    def __init__(self, channels, bias_final):
        super(MLP, self).__init__()
        self.n_layers = len(channels) - 1
        self.channels = channels
        bias = [1 for i in range(self.n_layers)]
        bias[-1] = bias_final
        for i in range(self.n_layers):
            self.add_link('l{}'.format(i), L.Linear(channels[i], channels[i+1], initial_bias=bias[i]))
            # self.add_link('l{}'.format(i), L.Linear(channels[i], channels[i+1]))

    def z(self, batch):
        return self.xp.random.randn(batch, self.channels[0]).astype('f')

    def __call__(self, x):
        for i in range(self.n_layers):
            x = self['l{}'.format(i)](x)
            if i + 1 == self.n_layers:
                x = F.relu(x)
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
    
def calc_lp(n, p, a, xp, f):
    EPS = 1e-6
    lp = F.sum(a * F.log(p + EPS) + (1 - a) * F.log(1 - p + EPS))

    return lp

def gen_initial_graph_state(G, n, s):
    post = [[] for _ in range(n*n)]
    dist = [10**6 for _ in range(n)]
    start_node = s
    for i in range(n):
        if G[start_node][i] != -1:
            post[start_node*n+i].append(int(G[start_node][i]))
    dist[start_node] = 0
    return post, dist

def decide_message(n, post):
    max_message = -1
    ind = -1
    for i in range(n*n):
        if post[i] != []:
            message = post[i][0]
            if message > max_message:
                max_message = message
                ind = i
    return ind

def update_state(n, G, dist, post):
    ind = decide_message(n, post)
    message = post[ind].pop(0)
    receive = ind%n
    if dist[receive] > message:
        dist[receive] = message
        for i in range(n):
            if G[receive][i] != -1:
                post[receive*n+i].append(dist[receive]+G[receive][i])
    return post, dist

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

    channels = [10, 100, 500, 1+n*(n-1)//2]
    if 'channels' in conf:
        channels = conf['channels']
        channels.append(n*(n-1)//2)

    # bias = - np.log(1.0 / conf['p']  - 1)
    bias = 1
    net = MLP(channels, bias)
    net.xp.random.seed(conf['seed'])

    if conf['gpu'] != -1:
        chainer.cuda.get_device_from_id(conf['gpu']).use()
        net.to_gpu()
    
    if conf['opt'] == 'SGD':
        opt = chainer.optimizers.SGD(lr=conf['lr'])
    elif conf['opt'] == 'Adam':
        opt = chainer.optimizers.Adam(alpha=conf['lr'])
    opt.setup(net)

    stop = 0

    pool_size = 10
    start_training = 20
    r_bests = []
    edges_bests = []
    z_bests = []

    if no_replay:
        pool_size = 1
        start_training = 1e9

    iteration = 0
    from_restart = 0

    start_time = time.time()

    # HiSampler Training
    while True:
        iteration += 1
        from_restart += 1

        z = net.z(1)
        x = net(z)[0].data
        G = net.xp.ones((n,n))*-1
        num_edges = 0
        count = 0
        start_candidate = set()
        for i in range(n-1):
            for j in range(i+1,n):
                G[i][j] = int(x[count])-1
                G[j][i] = G[i][j]
                count += 1
                if G[i][j] != -1:
                    num_edges += 1
                    start_candidate.add(i)
                    start_candidate.add(j)
        
        start_node = int(x[-1])
        if start_node not in start_candidate:
            start_node = net.xp.random.choice(net.xp.array(list(start_candidate)))
        post,dist = gen_initial_graph_state(G,n,start_node)
        check = True
        r = 0
        while check:
            post, dist = update_state(n, G, dist, post)
            # print(post)
            # time.sleep(1)
            r += 1
            check = False
            for p in post:
                if p != []:
                    check = True
                    break
        
        lp = F.sum(F.log(x + 1e-06))
        loss = - r * lp

        net.cleargrads()
        loss.backward()
        opt.update()

        entropy = F.mean(x * F.log(x + 1e-6) + (1 - x) * F.log(1 - x + 1e-6))

        # if no_replay:
        #     loss = - r * lp

        #     net.cleargrads()
        #     loss.backward()
        #     opt.update()

        if r > ma:
            ma = r
            stop = 0
        else:
            stop += 1
        if r > global_ma:
            global_ma = r
            output_graph(os.path.join(savedir, 'output_{}.txt'.format(r)), n, [G,num_edges,start_node], 3)
            # output_distribution_graph(os.path.join(savedir, 'distribution_{}.txt'.format(r)), n, x)
            # chainer.serializers.save_npz(os.path.join(savedir, 'snapshot_at_reward_{}'.format(r)), net)

        elapsed = time.time() - start_time

        ave = ave * (1 - conf['eps']) + r * conf['eps']
        aves.append(ave)
        with open(logfile, 'a') as f:
            print(savedir, iteration, elapsed, r, num_edges, entropy.data, global_ma, ma, ave, flush=True)
            print(iteration, elapsed, r, num_edges, entropy.data, global_ma, ma, ave, flush=True, file=f)

        plt.clf()
        plt.plot(range(len(aves)), aves)
        plt.savefig(os.path.join(savedir, 'graph.png'))

        # f = False
        # for es in edges_bests:
        #     if (es == inputs_li).all():
        #         f = True
        # if not f:
        #     r_bests.append(r)
        #     edges_bests.append(inputs_li)
        #     z_bests.append(z)
        
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

        # if stop >= conf['restart']:
        #     stop = 0
        #     ma = 0
        #     r_bests = []
        #     edges_bests = []
        #     z_bests = []
        #     from_restart = 0
        #     net = MLP(channels, bias)
        #     if conf['gpu'] != -1:
        #         chainer.cuda.get_device_from_id(conf['gpu']).use()
        #         net.to_gpu()
        #     if conf['opt'] == 'SGD':
        #         opt = chainer.optimizers.SGD(lr=conf['lr'])
        #     elif conf['opt'] == 'Adam':
        #         opt = chainer.optimizers.Adam(alpha=conf['lr'])
        #     opt.setup(net)
        #     continue

        # if iteration % 100 == 0:
        #     plt.clf()
        #     plt.plot(range(len(aves)), aves)
        #     plt.savefig(os.path.join(savedir, 'graph.png'))
        # if iteration % 1000 == 0:
        #     plt.savefig(os.path.join(savedir, 'graph_{}.png'.format(iteration)))
        #     plt.savefig(os.path.join(savedir, 'graph_{}.eps'.format(iteration)))
        #     chainer.serializers.save_npz(os.path.join(savedir, 'snapshot_{}'.format(iteration)), net)
        #     chainer.serializers.save_npz(os.path.join(savedir, 'opt_{}'.format(iteration)), opt)

if __name__ == '__main__':
    train()