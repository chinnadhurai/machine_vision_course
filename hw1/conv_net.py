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
from theano.tensor.signal.pool import pool_2d as max_pool_2d
import sys


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

        self.b1  = theano.shared(np.asarray(np.zeros((1,64,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))#l.init_weights((1,64,1,1))      #conv
        self.b2  = theano.shared(np.asarray(np.zeros((1,128,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))#l.init_weights((128,))    #conv
        self.b3  = theano.shared(np.asarray(np.zeros((1,256,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))#l.init_weights((256,))   #conv
        self.b4  = theano.shared(np.asarray(np.zeros((1,256,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))#l.init_weights((256,))   #conv
        self.b5  = theano.shared(np.asarray(np.zeros((1,1024,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))#l.init_weights((1024,))     #full-conn
        self.b6  = theano.shared(np.asarray(np.zeros((1,1024,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))#l.init_weights((1024,))    #full-conn
        self.b_o = l.init_weights((10,))      #full-conn

        #batch_norm params
        self.b =theano.shared(np.zeros((1,10)),broadcastable=(True,False))#l.init_weights((1,10))
        self.g =theano.shared(np.ones((1,10)),broadcastable=(True,False))#l.init_weights((1,10))
        self.r_m =theano.shared(np.zeros((1,10)),broadcastable=(True,False))#l.init_weights((1,10))
        self.r_s =theano.shared(np.zeros((1,10)),broadcastable=(True,False))#l.init_weights((1,10))
        print "Initializing and building conv_net"

    def bn(self, inputs, gamma, beta, mean, std):
            return T.nnet.bn.batch_normalization(inputs,
                                                 gamma,
                                                 beta,
                                                 mean,
                                                 std,
                                                 mode='low_mem')

    def model(self, X, w1, w2, w3, w4, w5, w6,w_o, p_drop_conv, p_drop_hidden):
        l1a = l.rectify(conv2d(X, w1, border_mode='valid') + self.b1)
        l1 = max_pool_2d(l1a, (2, 2), ignore_border=True)
        #l1 = l.dropout(l1, p_drop_conv)

        l2a = l.rectify(conv2d(l1, w2,border_mode='valid') + self.b2)
        l2 = max_pool_2d(l2a, (2, 2), ignore_border=True)
        #l2 = l.dropout(l2, p_drop_conv)

        l3 = l.rectify(conv2d(l2, w3, border_mode='valid') + self.b3)
        #l3 = l.dropout(l3a, p_drop_conv)

        l4a = l.rectify(conv2d(l3, w4, border_mode='valid') + self.b4)
        l4 = max_pool_2d(l4a, (2, 2), ignore_border=True)
        #l4 = T.flatten(l4, outdim=2)
        #l4 = l.dropout(l4, p_drop_conv)

        l5 = l.rectify(conv2d(l4, w5, border_mode='valid') + self.b5)
        #l5 = l.dropout(l5, p_drop_hidden)

        l6 = l.rectify(conv2d(l5, w6, border_mode='valid') + self.b6)
        #l6 = l.dropout(l6, p_drop_hidden)
        #l6 = self.bn(l6, self.g,self.b,self.m,self.v)
        l6 = conv2d(l6, w_o, border_mode='valid')
        #l6 = self.bn(l6, self.g, self.b, T.mean(l6, axis=1), T.std(l6,axis=1))
        l6 = T.flatten(l6, outdim=2)
        #l6 = ((l6 - T.mean(l6, axis=0))/T.std(l6,axis=0))*self.g + self.b#self.bn( l6, self.g,self.b,T.mean(l6, axis=0),T.std(l6,axis=0) )
        l6 = ((l6 - T.mean(l6, axis=0))/(T.std(l6,axis=0)+1e-4))*self.g + self.b
        pyx = T.nnet.softmax(l6)
        return l1, l2, l3, l4, l5, l6, pyx

    def update_running_mean_std(self, updates, i_m, i_s, a = 0.99):
        updates.append((self.r_m, a*self.r_m + (1-a)*i_m ))
        updates.append((self.r_s, a*self.r_s + (1-a)*i_s ))

    def test_model(self, X, w1, w2, w3, w4, w5, w6,w_o, p_drop_conv, p_drop_hidden):
        l1a = l.rectify(conv2d(X, w1, border_mode='valid') + self.b1)
        l1 = max_pool_2d(l1a, (2, 2), ignore_border=True)
        #l1 = l.dropout(l1, p_drop_conv)

        l2a = l.rectify(conv2d(l1, w2,border_mode='valid') + self.b2)
        l2 = max_pool_2d(l2a, (2, 2), ignore_border=True)
        #l2 = l.dropout(l2, p_drop_conv)

        l3 = l.rectify(conv2d(l2, w3, border_mode='valid') + self.b3)
        #l3 = l.dropout(l3a, p_drop_conv)

        l4a = l.rectify(conv2d(l3, w4, border_mode='valid') + self.b4)
        l4 = max_pool_2d(l4a, (2, 2), ignore_border=True)
        #l4 = T.flatten(l4, outdim=2)
        #l4 = l.dropout(l4, p_drop_conv)

        l5 = l.rectify(conv2d(l4, w5, border_mode='valid') + self.b5)
        #l5 = l.dropout(l5, p_drop_hidden)

        l6 = l.rectify(conv2d(l5, w6, border_mode='valid') + self.b6)
        #l6 = l.dropout(l6, p_drop_hidden)
        #l6 = self.bn(l6, self.g,self.b,self.m,self.v)
        l6 = conv2d(l6, w_o, border_mode='valid')
        #l6 = self.bn(l6, self.g, self.b, T.mean(l6, axis=1), T.std(l6,axis=1))
        l6 = T.flatten(l6, outdim=2)
        #l6 = ((l6 - T.mean(l6, axis=0))/T.std(l6,axis=0))*self.g + self.b#self.bn( l6, self.g,self.b,T.mean(l6, axis=0),T.std(l6,axis=0) )
        l6 = ((l6 - self.r_m)/(self.r_s + 1e-4))*self.g + self.b
        pyx = T.nnet.softmax(l6)
        return pyx

    def build_model(self):
        X, Y, w1, w2, w3, w4, w5, w6, w_o = self.X, self.Y, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w_o
        b1,b2,b3,b4,b5,b6 = self.b1,self.b2,self.b3,self.b4,self.b5,self.b6
        g, b = self.g, self.b
        l1, l2, l3, l4, l5, l6, py_x = self.model(X, w1, w2, w3, w4, w5, w6, w_o, 0., 0.)
        cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        params = [w1, w2, w3, w4, w5, w6, w_o, g, b, b1, b2, b3, b4, b5, b6]
        updates,grads = l.RMSprop(cost, params, lr=0.001)
        self.update_running_mean_std(updates, T.mean(l6, axis=0), T.std(l6,axis=0))
        self.train = theano.function(inputs=[X, Y], outputs=[cost,T.sum((grads)[0]),l1], updates=updates, allow_input_downcast=True)
        py_x = self.test_model(X, w1, w2, w3, w4, w5, w6, w_o, 0., 0.)
        y_x = T.argmax(py_x, axis=1)
        self.predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
        print "Done building the model..."

    def train(self):
        self.build_model()

        trX, trY, teX, teY = self.trX, self.trY, self.teX, self.teY
        mbsize = self.config['mini_batch_size']
        for i in range(10):
            print "epoch :",i
            for start, end in zip(range(0, len(trX), mbsize), range(mbsize, len(trX), mbsize)):
                #print start, trY[start:end].shape
                cost,grads,entropy = self.train(trX[start:end], trY[start:end])
                #y_x, l1, l2, l3, l4, l5, l6 = self.predict(trX[start:end])
                #print entropy.shape
                l.print_overwrite("cost : ",cost)
                #l.print_overwrite("gamma :",self.g.get_value()[0])
                #l.print_overwrite("running mean",  (self.r_m).get_value())
                #exit(0)
            print "\tvalidation accuracy:",np.mean(teY == self.predict(teX))


