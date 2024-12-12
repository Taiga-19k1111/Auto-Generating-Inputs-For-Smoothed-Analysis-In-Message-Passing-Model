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

conf = load_conf()
n = conf['n']

class DualNet(chainer.Chain):
    def __init__(self):
        super(DualNet, self).__init__()
        with self.init_scope():
            self.conv0 = L.Convolution2D(2,  48, 3, pad=1)
            self.conv1 = L.Convolution2D(48, 48, 3, pad=1)
            self.conv2 = L.Convolution2D(48, 48, 3, pad=1)
            self.conv3 = L.Convolution2D(48, 48, 3, pad=1)
            self.conv4 = L.Convolution2D(48, 48, 3, pad=1)

            self.bn0 = L.BatchNormalization(48)
            self.bn1 = L.BatchNormalization(48)
            self.bn2 = L.BatchNormalization(48)
            self.bn3 = L.BatchNormalization(48)
            self.bn4 = L.BatchNormalization(48)

            self.conv_p1 = L.Convolution2D(48, 2, 1)
            self.bn_p1   = L.BatchNormalization(2)
            self.fc_p2   = L.Linear(8 * 8 * 2, 8 * 8)

            self.conv_v1 = L.Convolution2D(48, 1, 1)
            self.bn_v1   = L.BatchNormalization(1)
            self.fc_v2   = L.Linear(8 * 8, 48)
            self.fc_v3   = L.Linear(48, 1)

    def __call__(self, x):
        # tiny ResNet
        h0 = F.relu(self.bn0(self.conv0(x)))
        h1 = F.relu(self.bn1(self.conv1(h0)))
        h2 = F.relu(self.bn2(self.conv2(h1)) + h0)
        h3 = F.relu(self.bn3(self.conv3(h2)))
        h4 = F.relu(self.bn4(self.conv4(h3)) + h2)

        h_p1 = F.relu(self.bn_p1(self.conv_p1(h4)))
        policy = self.fc_p2(h_p1)

        h_v1  = F.relu(self.bn_v1(self.conv_v1(h4)))
        h_v2  = F.relu(self.fc_v2(h_v1))
        value = F.tanh(self.fc_v3(h_v2))

        return policy, value
    
def get_dualnet_input_data(graph, post, xp):
    ini_post = xp.ones(n,n)

    x = xp.concatenate((graph, ini_post)).reshape((1, 4, 8, 8))
    return x