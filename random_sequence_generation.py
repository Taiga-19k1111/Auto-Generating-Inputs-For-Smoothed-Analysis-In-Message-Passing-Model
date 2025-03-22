#!/usr/bin/env python3

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

from util import calc_reward, makedir, output_sequence, output_distribution_sequence, load_conf

def gen_edges(n):
    EPS = 1e-6
    p = np.ones(n*(n-1)//2)*(1/2)
    a = np.random.binomial(1, p, n * (n-1) // 2)
    edges = np.array(np.tril_indices(n, -1)).T[np.where(a == 1)]
    return edges

def gen_sequence(n):
    EPS = 1e-6
    # sequence = []
    # for i in range(n):
    #     rnd = np.random.randint(0,n)
    #     sequence.append(rnd)
    sequence = np.random.permutation(n)
    return sequence

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

    iteration = 0
    from_restart = 0

    start_time = time.time()

    while iteration <= conf['restart']:
        iteration += 1
        from_restart += 1

        inputs = gen_sequence(n)

        r = calc_reward(n, inputs, solver, tmpdir, form)

        if r > ma:
            ma = r

        if r > global_ma:
            global_ma = r
            output_sequence(os.path.join(savedir, 'output_{}.txt'.format(r)), n, inputs)


        elapsed = time.time() - start_time

        ave = ave * (1 - conf['eps']) + r * conf['eps']
        aves.append(ave)
        with open(logfile, 'a') as f:
            # print(savedir, iteration, elapsed, r, len(inputs), global_ma, ma, ave, flush=True)
            print(iteration, elapsed, r, len(inputs), global_ma, ma, ave, flush=True, file=f)


        if iteration % 100 == 0:
            plt.clf()
            plt.plot(range(len(aves)), aves)
            plt.savefig(os.path.join(savedir, 'graph.png'))
        if iteration % 1000 == 0:
            plt.savefig(os.path.join(savedir, 'graph_{}.png'.format(iteration)))
            plt.savefig(os.path.join(savedir, 'graph_{}.eps'.format(iteration)))

if __name__ == '__main__':
    train()
