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
from load_data import load_cifar_10_data
import lib as l
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

class conv_net:
    def __init__(self, config):
        self.config = config
        self.trX, self.trY, self.teX, self.teY = load_cifar_10_data(config)
        self.X = T.ftensor4()
        self.Y = T.fmatrix()


        #weights init
        self.w   = l.init_weights((64, 3, 3, 3))      #conv
        self.w2  = l.init_weights((128, 64, 3, 3))    #conv
        self.w3  = l.init_weights((256, 128, 3, 3))   #conv
        self.w4  = l.init_weights((256, 256, 3, 3))   #conv
        self.w5  = l.init_weights((256*3*3, 1024))    #full-conn
        self.w6  = l.init_weights((1024, 1024))       #full-conn
        self.w_o = l.init_weights((1024, 10))         #full-conn

        #batch_norm params
        b = T.vector('b')
        g = T.vector('g')
        m = T.vector('m')
        v = T.vector('v')

        print "created conv_net"

    def model(X, w, w2, w3, w4, w5, w6,w_o, p_drop_conv, p_drop_hidden):
        l1a = l.rectify(conv2d(X, w, border_mode='full'))
        l1 = max_pool_2d(l1a, (2, 2))
        #l1 = l.dropout(l1, p_drop_conv)

        l2a = l.rectify(conv2d(l1, w2))
        l2 = max_pool_2d(l2a, (2, 2))
        #l2 = l.dropout(l2, p_drop_conv)

        l3a = l.rectify(conv2d(l2, w3))
        #l3 = l.dropout(l3a, p_drop_conv)

        l4a = l.rectify(conv2d(l3, w4))
        l4b = max_pool_2d(l4a, (2, 2))
        l4 = T.flatten(l4b, outdim=2)
        #l4 = l.dropout(l4, p_drop_conv)

        l5 = l.rectify(T.dot(l4, w5))
        #l5 = l.dropout(l5, p_drop_hidden)

        l6 = l.rectify(T.dot(l5, w6))
        #l6 = l.dropout(l6, p_drop_hidden)

        pyx = l.softmax(T.dot(l6, w_o))
        return l1, l2, l3, l4, pyx