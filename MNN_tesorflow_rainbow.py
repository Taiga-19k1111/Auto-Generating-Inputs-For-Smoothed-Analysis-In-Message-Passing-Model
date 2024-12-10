import os
import time

import matplotlib
matplotlib.use('Agg')

import numpy as np
import cupy as cp
import tensorflow as tf
import tensorflow.keras.layers as kl
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

class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, trainable=True):
        super(NoisyDense, self).__init__()
        self.units = units
        self.trainable = trainable
        self.activation = tf.keras.activation.get(activation)
        self.sigma_0 = 0.5

    def build(self, input_shape):
        p = input_shape[-1]
        self.w_mu = self.add_weight(
            name='w_mu', shape=(int(input_shape[-1]), self.units), initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(p), 1/np.sqrt(p)), trainable=self.trainable
        )

        self.w_sigma = self.add_weight(
            name='w_sigma', shape=(int(input_shape[-1]), self.units), initializer=tf.keras.initializers.Constant(self.sigma_0/np.sqrt(p)), trainable=self.trainable
        )

        self.b_mu = self.add_weight(
            name='b_mu', shape=(self.units,), initializer=tf.keras.initializers.RandomUniform(-1/np.sqrt(p),1/np.sqrt(p)), trainable=self.trainable
        )

        self.b_sigma = self.add_weight(
            name="b_sigma", shape=(self.units,), initializer=tf.keras.initializers.Constant(self.sigma_0/np.sqrt(p)), trainable=self.trainable
        )
    
    def call(self, inputs, noise=True):
        epsilon_in = self.f(tf.random.normal(shape=(self.w_mu.shape[0],1), dtype=tf.float32))
        epsilon_out = self.f(tf.random.normal(shape=(1,self.w_mu.shape[1]), dtype=tf.float32))

        w_epsilon = tf.matmul(epsilon_in, epsilon_out)
        b_epsilon = epsilon_out

        w = self.w_mu + self.w_sigma*w_epsilon
        b = self.b_mu + self.b_sigma*b_epsilon

        out = tf.matmul(inputs,w) + b
        if self.activation is not None:
            out = self.activation(out)

        return out

    @staticmethod
    def f(x):
        x = tf.sign(x)*tf.sqrt(tf.abs(x))
        return x

class RainbowQNetwork(tf.keras.Model):
    def __init__(self, action_space, Vmin, Vmax, n_atoms):
        super(RainbowQNetwork, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        self.Vmin, self.Vmax = Vmin, Vmax
        self.Z = np.linspace(self.Vmin, self.Vmax, self.n_atoms)
        self.conv1 = kl.Conv2D(32,8,stride=4,activation="relu",kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64,4,stride=2,activation="relu",kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64,3,stride=1,activation="relu",kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()
        self.dense1 = NoisyDense(512, activation="relu")
        self.dense2 = NoisyDense(512, activation="relu")
        self.value = NoisyDense(1*self.n_atoms)
        self.advantages = NoisyDense(self.action_space*self.n_atoms)

    @tf.function
    def call(self, x):
        batch_size = x.shape[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten1(x)

        x1 = self.dense1(x)
        value = self.value(x1)
        value = tf.reshape(value, (batch_size, 1, self.n_atoms))

        x2 = self.dense2(x)
        advantages = tf.reshape(advantages, (batch_size, self.acton_space, self.n_atoms))
        advantages_mean = tf.reduce_mean(advantages, axis=1, keepdims=True)
        advantages_scaled = advantages - advantages_mean
        logits = value + advantages_scaled
        probs = tf.nn.softmax(logits, axis=2)

        return probs

    def sample_action(self, x):
        selected_actions, _ = self.sample_actions(x)
        selected_action = selected_actions[0][0].numpy()
        return selected_action
    
    def sample_actions(self, X):
        probs = self(X)
        q_means = tf.reduce_sum(probs*self.Z, axis=2, keepdims=True)
        selected_actions = tf.argmax(q_means, axis=1)
        return selected_actions, probs    

class Memory():
    def __init__(self, memory_size=100000):
        self.buffer = deque(maxlen=memory_size)
    
    def add(self, experience):
        self.buffer.append(experience)

    def sample(self,batch_size):
        idx = cp.random.choice(cp.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[int(i)] for i in cp.asnumpy(idx)]
    
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

    return cp.array(G.ravel()),post

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
    E_DECAY_RATE = 0.00001
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

            if epsilon > cp.random.uniform(0,1,1):
                mx = cp.random.uniform(0,1,n*n)
                ep_check = 0
            else:
                cmnn += 1
                mx = main_qn.model.predict(cp.asnumpy([G[n*n:n*n*2]]))[0]
                ep_check = 1

            if ep%100 == 0:
                output_distribution_graph(os.path.join(savedir, 'distribution_{}_{}_{}.txt'.format(ep,step,ep_check)), n, mx)

            inputs = decide_message(n,mx,G,post)
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
                inputs = cp.zeros([pool_size,n*n])
                targets = cp.zeros([pool_size,n*n])

                minibatch = memory.sample(pool_size)
                for i, [G_b,inputs_b,reward_b,next_G_b] in enumerate(minibatch):
                    inputs[i] = G_b[n*n:n*n*2]
                    next_initial_post = (next_G_b[n*n:n*n*2] != -1)
                    if next_initial_post.any():
                        next_q = target_qn.model.predict(cp.asnumpy([next_G_b[n*n:n*n*2]]))[0]
                        max_q = cp.amax(next_q*cp.where(next_initial_post,1,0))
                        target = reward_b + GAMMA*max_q
                    else:
                        target =reward_b
                    targets[i] = main_qn.model.predict(cp.asnumpy([inputs[i]]))[0]
                    targets[i][inputs_b] = target
                main_qn.model.fit(cp.asnumpy(inputs), cp.asnumpy(targets), epochs=1, verbose=0)

        memo_x.append(ep)
        memo_y.append(step)
        output_graph(os.path.join(savedir, 'output_{}.txt'.format(ep)), n, edges, 0)
        plt.clf()
        plt.plot(memo_x, memo_y)
        plt.savefig(os.path.join(savedir, 'graph.png'))

        print(epsilon, np.mean(memo_y), cmnn)

if __name__ == '__main__':
    train()