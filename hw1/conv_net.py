__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image
import cPickle as pickle
from load_data import load_data
import lib as l

class conv_net:
    def __init__(self, config):
        self.config = config
        self.data = load_data(config)
        print "created conv_net"

    def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
        l1a = l.rectify(l.conv2d(X, w, border_mode='full'))
        l1 = l.max_pool_2d(l1a, (2, 2))
        l1 = l.dropout(l1, p_drop_conv)

        l2a = l.rectify(l.conv2d(l1, w2))
        l2 = l.max_pool_2d(l2a, (2, 2))
        l2 = l.dropout(l2, p_drop_conv)

        l3a = l.rectify(l.conv2d(l2, w3))
        l3b = l.max_pool_2d(l3a, (2, 2))
        l3 = l.T.flatten(l3b, outdim=2)
        l3 = l.dropout(l3, p_drop_conv)

        l4 = l.rectify(T.dot(l3, w4))
        l4 = l.dropout(l4, p_drop_hidden)

        pyx = l.softmax(T.dot(l4, w_o))
        return l1, l2, l3, l4, pyx