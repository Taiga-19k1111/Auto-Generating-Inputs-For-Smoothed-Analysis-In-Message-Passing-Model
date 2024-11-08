import os
import subprocess
import argparse

import numpy as np
import yaml

def get_output(command, form):
    if form == 2:
        o = subprocess.check_output(command, shell=True).decode().strip().split('\n')
        return o
    else:
        return subprocess.check_output(command, shell=True).decode()
    
def output_graph(filename, n, inputs, form):
    with open(filename, 'w') as f:
        if form == 0:
            f.write('{} {}\n'.format(n, len(inputs)))
            for e in inputs:
                f.write('{} {}\n'.format(e[0], e[1]))
        elif form == 2:
            send,receive,G,_ = inputs
            f.write('{} {} {}\n'.format(n, send, receive))
            for i in range((n*2)+1):
                for j in range(n):
                    if j != 0:
                        f.write(' ')
                    f.write('{}'.format(G[i*n+j]))
                f.write('\n')                    

def output_sequence(filename, n , sequence):
    with open(filename, 'w') as f:
        f.write('{} {}\n'.format(n, len(sequence)))
        for i in range(n):
            f.write('{} '.format(sequence[i]))   

def output_distribution_graph(filename, n, x):
    P = np.zeros((n, n), dtype='f')
    cnt = 0
    for i in range(n):
        for j in range(n):
            P[i, j] = x[cnt]
            cnt += 1
    with open(filename, 'w') as f:
        f.write('{}\n'.format(n))
        for i in range(n):
            for j in range(n):
                if j != 0:
                    f.write(' ')
                f.write('{:.3f}'.format(P[i, j]))
            f.write('\n')

def output_distribution_sequence(filename, n, x):
    x = x.reshape(n,n)
    with open(filename, 'w') as f:
        f.write('{}\n'.format(n))
        for i in range(n):
            for j in range(n):
                if j != 0:
                    f.write(' ')
                f.write('{:.3f}'.format(x[i, j].item()))
            f.write('\n')

    # with open(filename, 'w') as f:
    #     f.write('{}\n'.format(n))
    #     for i in range(n):
    #         if i != 0:
    #             f.write(' ')
    #         f.write('{:.3f}'.format(x[i].item()))
    #         f.write('\n')    

def update_state(inputs,output,n):
    post = inputs[3]
    G = inputs[2]
    r = 0
    count = 0
    if output[0] != '':
        send,memory = list(map(int,output.pop(0).split()))
        for o in output:
            receive,message = list(map(int,o.split()))
            post[send*n+receive].append(message)
            r += message
            count += 1
            # r = r+1
        # r = 1/(G[-n+send]-memory)
        r = G[-n+send]-memory
        # r = memory
        # r = r/count
        G[-n+send] = memory

    for i in range(n*n):
        if post[i] == []:
            G[i+n*n] = -1
        else:
            G[i+n*n] = post[i][0]
        
    return post, G, r

def gen_random_graph(n,p,max_m):
    post = [[] for _ in range(n*n)]
    edges = np.random.binomial(1,p,n*(n-1)//2)
    message = np.random.binomial(1,p,(n,n))
    contents = np.random.randint(0,max_m,(n,n))
    weight = np.random.randint(0,max_m,n*(n-1)//2)
    graph = np.zeros((n,n))
    ini_post = np.ones([n,n])*-1
    count = 0
    for i in range(n-1):
        for j in range(i+1,n):
            if edges[count] == 1:
                graph[i][j] = weight[count]
                graph[j][i] = weight[count]
                if message[i][j] == 1:
                    ini_post[i][j] = contents[i][j]
                    post[i*n+j].append(message[i][j])
                if message[j][i] == 1:
                    ini_post[j][i] = contents[j][i]
                    post[j*n+i].append(contents[j][i])
            count += 1
    node_memory = np.random.randint(0,max_m,n)
    return np.concatenate([graph.ravel(),ini_post.ravel(),node_memory]).ravel(),post

def calc_reward(n, inputs, solver, tmpdir, form):
    filename = os.path.join(tmpdir, 'calc_reward_{}'.format(os.getpid()))
    if form == 0 or form == 2:
        output_graph(filename, n, inputs, form)
    elif form == 1:
        output_sequence(filename, n, inputs)

    if form == 2:
        output = get_output("{} < {}".format(solver, filename), form)
        new_post, new_G, reward = update_state(inputs,output,n)
        # reward = 0
        # for p in new_post:
        #     reward += len(p)
        return new_post, new_G, reward
    else:
        reward = float(get_output("{} < {}".format(solver, filename), form))
        os.remove(filename)
        return reward 

def makedir(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def load_conf():
    parser = argparse.ArgumentParser()
    parser.add_argument('conf', type=str)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--solver', type=str, default=None)
    parser.add_argument('--gpu', '-g', type=int, default=None)
    parser.add_argument('--form', type=int, default=None)
    parser.add_argument('--n', type=int, default=None)
    parser.add_argument('--p', type=float, default=None)
    parser.add_argument('--iteration', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--dirname', type=str, default=None)
    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--restart', type=int, default=None)
    parser.add_argument('--noreplay', action='store_true')
    parser.add_argument('--opt', type=str, default='Adam')
    parser.add_argument('--epoch', type=int, default=None)
    parser.add_argument('--erp', type=float, default=None)
    parser.add_argument('--message', type=int, default=None)
    args = parser.parse_args()

    with open(args.conf) as f:
        conf = yaml.safe_load(f)

    for key in conf:
        conf[key] = getattr(args, key) if getattr(args, key) is not None else conf[key]

    return conf
