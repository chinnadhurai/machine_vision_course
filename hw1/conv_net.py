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
from theano.tensor.nnet import conv2d
from theano.tensor.signal.downsample import max_pool_2d


class conv_net:
    def __init__(self, config):
        self.config = config
        print "Experiment Configuration:"
        print "Num training examples    : ", config["ntrain"]
        print "Num test examples        : ", config["ntest"]
        print "Minibatch size           : ", config["mini_batch_size"]
        self.trX, self.trY, self.teX, self.teY = load_cifar_10_data(config)
        self.X = T.ftensor4()
        self.Y = T.fmatrix()


        #weights init ( output depth/filers x input depth x filter_h x filter_w
        self.w1  = l.init_weights((64, 3, 3, 3))      #conv
        self.w2  = l.init_weights((128, 64, 3, 3))    #conv
        self.w3  = l.init_weights((256, 128, 3, 3))   #conv
        self.w4  = l.init_weights((256, 256, 3, 3))   #conv
        self.w5  = l.init_weights((1024,256,1,1))     #full-conn
        self.w6  = l.init_weights((1024,1024,1,1))    #full-conn
        self.w_o = l.init_weights((10,1024,1,1))      #full-conn

        self.b1  = l.init_weights((64,))      #conv
        self.b2  = l.init_weights((128,))    #conv
        self.b3  = l.init_weights((256,))   #conv
        self.b4  = l.init_weights((256,))   #conv
        self.b5  = l.init_weights((1024,))     #full-conn
        self.b6  = l.init_weights((1024,))    #full-conn
        self.b_o = l.init_weights((10,))      #full-conn

        #batch_norm params
        self.b = l.init_weights((1,1024))
        self.g = l.init_weights((1,1024))
        self.m = l.init_weights((1,1024))
        self.v = l.init_weights((1,1024))

        print "Initializing and building conv_net"

    def bn(self, inputs, gamma, beta, mean, std):
            return T.nnet.bn.batch_normalization(inputs,
                                                 gamma,
                                                 beta,
                                                 mean,
                                                 std,
                                                 mode='low_mem')

    def model(self, X, w1, w2, w3, w4, w5, w6,w_o, p_drop_conv, p_drop_hidden):
        l1a = l.rectify(conv2d(X, w1, border_mode='valid'))
        l1 = max_pool_2d(l1a, (2, 2), ignore_border=True)
        #l1 = l.dropout(l1, p_drop_conv)

        l2a = l.rectify(conv2d(l1, w2,border_mode='valid'))
        l2 = max_pool_2d(l2a, (2, 2), ignore_border=True)
        #l2 = l.dropout(l2, p_drop_conv)

        l3 = l.rectify(conv2d(l2, w3, border_mode='valid'))
        #l3 = l.dropout(l3a, p_drop_conv)

        l4a = l.rectify(conv2d(l3, w4, border_mode='valid'))
        l4 = max_pool_2d(l4a, (2, 2), ignore_border=True)
        #l4 = T.flatten(l4, outdim=2)
        #l4 = l.dropout(l4, p_drop_conv)

        l5 = l.rectify(conv2d(l4, w5, border_mode='valid'))
        #l5 = l.dropout(l5, p_drop_hidden)

        l6 = l.rectify(conv2d(l5, w6, border_mode='valid'))
        #l6 = l.dropout(l6, p_drop_hidden)
        #l6 = self.bn(l6, self.g,self.b,self.m,self.v)
        l6 = conv2d(l6, w_o, border_mode='valid')
        l6 = T.flatten(l6, outdim=2)
        pyx = T.nnet.softmax(l6)
        return l1, l2, l3, l4, l5, l6, pyx

    def build_model(self):
        X, Y, w1, w2, w3, w4, w5, w6, w_o = self.X, self.Y, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w_o
        l1, l2, l3, l4, l5, l6, py_x = self.model(X, w1, w2, w3, w4, w5, w6, w_o, 0., 0.)
        y_x = T.argmax(py_x, axis=1)
        cost = T.mean(T.nnet.categorical_crossentropy(Y,py_x))
        params = [w1, w2, w3, w4, w5, w6, w_o]
        updates,grads = l.RMSprop(cost, params, lr=0.001)

        self.train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
        self.predict = theano.function(inputs=[X], outputs=[l1, l2, l3, l4, l5, l6], allow_input_downcast=True)
        print "Done building the model..."

    def train(self):
        self.build_model()

        trX, trY, teX, teY = self.trX, self.trY, self.teX, self.teY
        mbsize = self.config['mini_batch_size']
        for i in range(2):
            for start, end in zip(range(0, len(trX), mbsize), range(mbsize, len(trX), mbsize)):
                print start, trY[start:end].shape
                cost = self.train(trX[start:end], trY[start:end])
                l1, l2, l3, l4, l5, l6 = self.predict(trX[start:end])
                print l1.shape, l2.shape, l3.shape, l4.shape, l5.shape, l6.shape

                exit(0)
            print "epoch :",i


