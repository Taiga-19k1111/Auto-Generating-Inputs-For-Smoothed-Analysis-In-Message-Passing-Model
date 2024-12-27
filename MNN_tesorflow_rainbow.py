import os
import time

import matplotlib
matplotlib.use('Agg')

import numpy as np
import cupy as cp
import copy
import tensorflow as tf
import tensorflow.keras.layers as kl
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from collections import deque
from tensorflow.keras.losses import Huber

import matplotlib.pyplot as plt
import yaml

from dataclasses import dataclass
import collections

import pickle
import zlib

from util import calc_reward, makedir, output_graph, output_distribution_graph, gen_random_graph, load_conf
from gen_SSSP_worstcase import gen_worstcase

class RainbowAgent:
    def __init__(self, n):
        self.is_noisy = False
        
        self.n = n
        self.gamma = 0.99
        self.batch_size = 32
        self.n_frames = 4
        self.update_period = 1
        self.target_update_period = 2000

        self.n_atoms = 51
        self.Vmin, self.Vmax = -10, 10
        self.delta_z = (self.Vmax - self.Vmin)/(self.n_atoms - 1)
        self.Z = np.linspace(self.Vmin, self.Vmax, self.n_atoms)

        self.n_step_return = 3

        self.action_space = self.n**2
        self.qnet = RainbowQNetwork(self.action_space, Vmin=self.Vmin, Vmax=self.Vmax, n_atoms=self.n_atoms, is_noisy=self.is_noisy)
        self.target_qnet = RainbowQNetwork(self.action_space, Vmin=self.Vmin, Vmax=self.Vmax, n_atoms=self.n_atoms, is_noisy=self.is_noisy)

        self.optimizer = tf.keras.optimizers.Adam(lr=0.0001, epsilon=0.01/self.batch_size)

        self.replay_buffer = NstepPrioritizedReplayBuffer(max_len=1000000, reward_clip=True, alpha=0.8, beta=0.4, total_steps=2500000, nstep_return=self.n_step_return, gamma=self.gamma)

        self.steps = 0

    def learn(self, n_episodes):
        E_START = 1.0
        E_STOP = 0
        E_DECAY_RATE = 0.0001
        GAMMA = 0.99
        memo_x = []
        memo_y = []
        memo_ave = []
        ave = 0
        total_max = 0
        initial_G = gen_worstcase(self.n)
        initial_G, initial_post = gen_initial_graph_state(initial_G, self.n)
        for ep in range(1,n_episodes+1):
            G = np.copy(initial_G)
            post = copy.deepcopy(initial_post)
            frame = G[self.n*self.n:self.n*self.n*2].reshape((self.n,self.n))
            frames = collections.deque([frame]*self.n_frames, maxlen=self.n_frames)
            edges = []
            check = True
            num_messages = 0
            while check:
                num_messages += 1
                self.steps += 1
                state = np.stack(frames, axis=2)[np.newaxis, ...]
                mask_post = np.where(G[self.n**2:(self.n**2)*2] == -1, 0, 1).reshape(1,self.n**2)
                if self.is_noisy:
                    action = self.qnet.sample_action(state, mask_post)
                else:
                    epsilon = E_STOP+(E_START-E_STOP)*np.exp(-E_DECAY_RATE*self.steps)
                    print(epsilon)
                    if epsilon > np.random.uniform(0,1,1):
                        mx = np.random.uniform(1,2,self.n**2)*mask_post.reshape(self.n**2)
                        action = np.argmax(mx)
                    else:
                        action = self.qnet.sample_action(state, mask_post)
                post[action].pop(0)
                inputs = [action//self.n, action%self.n, G, post]
                post, next_G, r = calc_reward(n, inputs, solver, tmpdir, form)
                next_frame = next_G[self.n**2:(self.n**2)*2].reshape((self.n,self.n))
                frames.append(next_frame)
                next_state = np.stack(frames, axis=2)[np.newaxis, ...]
                edges.append(inputs[0:2])
                G = next_G
                next_mask_post = np.where(G[self.n**2:(self.n**2)*2] == -1, 0, 1).reshape(1,self.n**2)

                check = False
                for po in post:
                    if po != []:
                        check = True
                        break

                if check:
                    reward = 0
                    transition = (state, action, reward, next_state, False, next_mask_post)
                else:
                    reward = 1-1/num_messages
                    transition = (state, action, reward, next_state, True, next_mask_post)
                
                self.replay_buffer.push(transition)

                if len(self.replay_buffer) >= 1000:
                    if self.steps%self.update_period == 0:
                        loss = self.update_network()
                        # print(loss)
                    
                    if self.steps%self.target_update_period == 0:
                        self.target_qnet.set_weights(self.qnet.get_weights())

            memo_x.append(ep)
            memo_y.append(num_messages)
            ave = ((ep-1)*ave+num_messages)/ep
            memo_ave.append(ave)
            plt.clf()
            plt.plot(memo_x, memo_y)
            plt.savefig(os.path.join(savedir, 'message_num.png'))
            plt.clf()
            plt.plot(memo_x, memo_ave)
            plt.savefig(os.path.join(savedir, 'ave_message_num.png'))

            if num_messages > total_max:
                total_max = num_messages
                output_graph(os.path.join(savedir, 'output_{}.txt'.format(total_max)), n, edges, 0)
    
    def update_network(self):
        indices, weights, (states, actions, rewards, next_states, dones, next_mask_post) = self.replay_buffer.get_minibatch(self.batch_size, self.steps)

        next_actions, _ = self.qnet.sample_actions(next_states, next_mask_post)
        _, next_probs = self.target_qnet.sample_actions(next_states, next_mask_post)

        onehot_mask = self.create_mask(next_actions)
        next_dists = tf.reduce_sum(next_probs*onehot_mask, axis=1).numpy()

        target_dists = self.shift_and_projection(rewards, dones, next_dists)

        onehot_mask = self.create_mask(actions)
        with tf.GradientTape() as tape:
            probs = self.qnet(states)
            dists = tf.reduce_sum(probs*onehot_mask, axis=1)

            dists = tf.clip_by_value(dists, 1e-6, 1.0)
            td_loss = tf.reduce_sum(-1*target_dists*tf.math.log(dists), axis=1, keepdims=True)

            weighted_loss = weights*td_loss
            loss = tf.reduce_mean(weighted_loss)

        grads = tape.gradient(loss, self.qnet.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.qnet.trainable_variables))

        td_loss = td_loss.numpy().flatten()
        self.replay_buffer.update_priority(indices, td_loss)

        return loss

    def shift_and_projection(self, rewards, dones, next_dists):
        target_dists = np.zeros((self.batch_size, self.n_atoms))
        for j in range(self.n_atoms):
            tZ_j = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards + (self.gamma**(self.n_step_return))*self.Z[j]))
            bj = (tZ_j - self.Vmin)/self.delta_z

            lower_bj = np.floor(bj).astype(np.int8)
            upper_bj = np.ceil(bj).astype(np.int8)

            eq_mask = lower_bj == upper_bj
            neq_mask = lower_bj != upper_bj

            lower_probs = 1 - (bj - lower_bj)
            upper_probs = 1 - (upper_bj - bj)

            next_dist = next_dists[:, [j]]
            indices = np.arange(self.batch_size).reshape(-1,1)

            target_dists[indices[neq_mask], lower_bj[neq_mask]] += (lower_probs*next_dist)[neq_mask]
            target_dists[indices[neq_mask], upper_bj[neq_mask]] += (upper_probs*next_dist)[neq_mask]

            target_dists[indices[eq_mask], lower_bj[eq_mask]] += (0.5*next_dist)[eq_mask]
            target_dists[indices[eq_mask], upper_bj[eq_mask]] += (0.5*next_dist)[eq_mask]

        for batch_idx in range(self.batch_size):
            if not dones[batch_idx]:
                continue
            else:
                target_dists[batch_idx, :] = 0
                tZ = np.minimum(self.Vmax, np.maximum(self.Vmin, rewards[batch_idx]))
                bj = (tZ - self.Vmin)/self.delta_z

                lower_bj = np.floor(bj).astype(np.int32)
                upper_bj = np.ceil(bj).astype(np.int32)

                if lower_bj == upper_bj:
                    target_dists[batch_idx, lower_bj] += 1.0
                else:
                    target_dists[batch_idx, lower_bj] += 1 - (bj - lower_bj)
                    target_dists[batch_idx, upper_bj] += 1 - (upper_bj - bj)

        return target_dists
    
    def create_mask(self, actions):
        mask = np.ones((self.batch_size, self.action_space, self.n_atoms))
        action_onehot = tf.one_hot(tf.cast(actions, tf.int32), self.action_space, axis=1)
        for idx in range(self.batch_size):
            mask[idx, ...] = mask[idx, ...]*action_onehot[idx, ...]

        return mask    

class NoisyDense(tf.keras.layers.Layer):
    def __init__(self, units, activation=None, trainable=True):
        super(NoisyDense, self).__init__()
        self.units = units
        self.trainable = trainable
        # self.activation = tf.keras.activation.get(activation)
        self.activation = tf.keras.layers.ReLU()
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
    def __init__(self, action_space, Vmin, Vmax, n_atoms, is_noisy):
        super(RainbowQNetwork, self).__init__()
        self.action_space = action_space
        self.n_atoms = n_atoms
        self.Vmin, self.Vmax = Vmin, Vmax
        self.Z = np.linspace(self.Vmin, self.Vmax, self.n_atoms)
        self.conv1 = kl.Conv2D(32,8,strides=4,padding='same',activation="relu",kernel_initializer="he_normal")
        self.conv2 = kl.Conv2D(64,4,strides=2,padding='same',activation="relu",kernel_initializer="he_normal")
        self.conv3 = kl.Conv2D(64,3,strides=1,padding='same',activation="relu",kernel_initializer="he_normal")
        self.flatten1 = kl.Flatten()
        if is_noisy:
            self.dense1 = NoisyDense(256, activation="relu")
            self.dense2 = NoisyDense(256, activation="relu")
            self.value = NoisyDense(1*self.n_atoms)
            self.advantages = NoisyDense(self.action_space*self.n_atoms)
        else:
            self.dense1 = kl.Dense(256, activation="relu")
            self.dense2 = kl.Dense(256, activation="relu")
            self.value = kl.Dense(1*self.n_atoms)
            self.advantages = kl.Dense(self.action_space*self.n_atoms)

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
        advantages = self.advantages(x2)
        advantages = tf.reshape(advantages, (batch_size, self.action_space, self.n_atoms))
        advantages_mean = tf.reduce_mean(advantages, axis=1, keepdims=True)
        advantages_scaled = advantages - advantages_mean
        logits = value + advantages_scaled
        probs = tf.nn.softmax(logits, axis=2)

        return probs

    def sample_action(self, x, masks):
        selected_actions, _ = self.sample_actions(x, masks)
        # selected_action = selected_actions[0][0].numpy()
        selected_action = selected_actions[0][0]
        return selected_action
    
    def sample_actions(self, X, masks):
        probs = self(X)
        # q_means = tf.reduce_sum(probs*self.Z, axis=2, keepdims=True)
        q_means = tf.reduce_sum(probs*self.Z, axis=2, keepdims=True).numpy()
        batch_size = q_means.shape[0]
        selected_actions = np.zeros((batch_size,1), dtype=np.int64)
        for idx in range(batch_size):
            mask = masks[idx]
            if not mask.any():
                mask = np.ones(mask.shape)
            post_in_message = np.where(mask == 1)[0]
            q_means_idx = q_means[idx,post_in_message,:]
            selected_actions[idx][0] = post_in_message[np.argmax(q_means_idx, axis=0)[0]]
        return selected_actions, probs

@dataclass
class Experience:
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool
    next_mask_post: np.ndarray

class NstepPrioritizedReplayBuffer:
    def __init__(self, max_len, gamma, reward_clip, nstep_return=3, alpha=0.6, beta=0.4, total_steps=2500000):
        self.max_len = max_len
        self.gamma = gamma
        self.buffer = []
        self.priorities = []

        self.nstep_return = nstep_return
        self.temp_buffer = collections.deque(maxlen=nstep_return)

        self.alpha = alpha
        self.beta_scheduler = (lambda steps: beta + (1 - beta)*steps/total_steps)
        self.epsilon = 0.01
        self.max_priority = 1.0

        self.reward_clip = reward_clip
        self.counter = 0

    def __len__(self):
        return len(self.buffer)
    
    def push(self, transition):
        self.temp_buffer.append(Experience(*transition))

        if len(self.temp_buffer) == self.nstep_return:
            nstep_return = 0
            has_done = False
            for i, exp in enumerate(self.temp_buffer):
                reward, done = exp.reward, exp.done
                reward = np.clip(reward, -1, 1) if self.reward_clip else reward
                nstep_return += (self.gamma**i)*(1 - done)*reward
                if done:
                    has_done = True
                    break
            
            nstep_exp = Experience(self.temp_buffer[0].state, self.temp_buffer[0].action, nstep_return, self.temp_buffer[-1].next_state, has_done, self.temp_buffer[-1].next_mask_post)
            nstep_exp = zlib.compress(pickle.dumps(nstep_exp))

            if self.counter == self.max_len:
                self.counter = 0
            
            try:
                self.buffer[self.counter] = nstep_exp
                self.priorities[self.counter] = self.max_priority
            except IndexError:
                self.buffer.append(nstep_exp)
                self.priorities.append(self.max_priority)

            self.counter += 1

    def get_minibatch(self, batch_size, steps):
        probs = np.array(self.priorities)/sum(self.priorities)
        indices = np.random.choice(np.arange(len(self.buffer)), p=probs, replace=False, size=batch_size)

        beta = self.beta_scheduler(steps)
        weights = (probs[indices]*len(self.buffer))**(-1*beta)
        weights /= weights.max()
        weights = weights.reshape(-1,1).astype(np.float32)

        selected_experiences = [pickle.loads(zlib.decompress(self.buffer[idx])) for idx in indices]

        states = np.vstack([exp.state for exp in selected_experiences]).astype(np.float32)
        actions = np.vstack([exp.action for exp in selected_experiences]).astype(np.float32)
        rewards = np.array([exp.reward for exp in selected_experiences]).reshape(-1,1)
        next_states = np.vstack([exp.next_state for exp in selected_experiences]).astype(np.float32)
        dones = np.array([exp.done for exp in selected_experiences]).reshape(-1,1)
        next_post_masks = np.vstack([exp.next_mask_post for exp in selected_experiences]).astype(np.int8)

        return indices, weights, (states, actions, rewards, next_states, dones, next_post_masks)
    
    def update_priority(self, indices, td_errors):
        priorities = (np.abs(td_errors) + self.epsilon)**self.alpha
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        self.max_priority = max(self.max_priority, priorities.max())

    
def gen_initial_graph_state(g, n):
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

    return np.array(G.ravel()),post

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

if __name__ == '__main__':
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
    cp.random.seed(conf['seed'])
    tf.random.set_seed(conf['seed'])

    logfile = os.path.join(savedir, 'log')
    
    RA = RainbowAgent(n)
    RA.learn(100000)
