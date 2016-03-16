__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image
import cPickle as pickle
from load_data import load_cifar_10_data,load_cifar_100_data
import lib as l
#from theano.tensor.nnet import conv2d
from theano.sandbox.cuda.dnn import dnn_conv as conv2d
from theano.tensor.signal.pool import pool_2d as max_pool_2d
import sys
from scipy.misc import imread
import vgg_16
import lasagne


class conv_classifier_type:
    def __init__(self, config):
        self.config = config
        self.X = T.fmatrix()
        self.Y = T.fmatrix()
        self.params = self.load_params()
        self.trX, self.trY, self.teX, self.teY = load_cifar_10_data(config)          "Initialized conv_classifier"

    def load_params(self):
        param_loc = self.config['params']
        params = l.load_params_pickle(param_loc)
        print "Loaded params from ", param_loc      
        return params['param values']
    
    def build_model(self, input_var=None):
        l_in = lasagne.layers.InputLayer(shape=(None, 1, 1000),
                                         input_var=input_var)
        l_out = lasagne.layers.DenseLayer( 
                l_in, 
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)
        return l_out

    def compile_model(self):
        X,Y = self.X,self.Y
        network = build_model(X)
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
        
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
        train = theano.function([X, Y], loss, updates=updates)      
        predict = theano.function([X, Y], [test_prediction])
        "Compiled model..."
        return train,predict

    def train(self):
        X = self.X
        self.net = vgg_16.build_model()  
        lasagne.layers.set_all_param_values(self.net['prob'], self.params)
        train,predict = self.compile_model()
               





           
