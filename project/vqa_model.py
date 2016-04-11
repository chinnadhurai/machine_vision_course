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
import load_data as load_data
import lib as l
#from theano.tensor.nnet import conv2d
from theano.sandbox.cuda.dnn import dnn_conv as conv2d
from theano.tensor.signal.pool import pool_2d as max_pool_2d
import sys
from scipy.misc import imread
import vgg_16
import lasagne
from lasagne.regularization import regularize_layer_params, l2,l1
import os
import re
from lasagne.layers import *
from pprint import pprint

class vqa_type:
    def __init__(self, config):
        print "Configuration :"
        self.config                     = config
        self.qX                         = T.ftensor3()
        self.lstm_mask                  = T.imatrix()
        self.iX                         = T.fmatrix()
        self.Y                          = T.ivector()
        self.mlp_input_dim              = 1024
        self.q_embed_dim                = 300
        self.num_answers                = 18
        self.bptt_trunk_steps           = -1
        self.grad_clip                  = 100
        self.batch_size                 = 128
        self.max_seq_length             = 10
        pprint(config)
        print "\n----------------------"
        print "Initialization done ..."
    
    def build_question_lstm(self, input_var, mask=None):
        input_dim, seq_len, mb_size = self.q_embed_dim, self.max_seq_length, self.batch_size
        # (batch size, max sequence length, number of features)
        l_in = lasagne.layers.InputLayer(shape=(mb_size, seq_len, input_dim),
                                            input_var=input_var)
        l_mask = lasagne.layers.InputLayer(shape=(mb_size, seq_len), input_var=mask)
        l_lstm = lasagne.layers.LSTMLayer(l_in, 
                                          num_units             = self.q_embed_dim, 
                                          only_return_final     = True,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          grad_clipping         = self.grad_clip,
                                          mask_input            = l_mask
                                         )
        l_dense = lasagne.layers.DenseLayer(l_lstm, num_units=self.mlp_input_dim)
        net  = {'l_in':l_in, 'l_lstm':l_lstm, 'l_dense':l_dense}
        print "Done building question LSTM ..."
        return net
    
    def combine_image_question_model(self,image_feature, question_feature):
        return image_feature * question_feature
    
    def build_mlp_model(self,input_var):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,self.mlp_input_dim),
                                         input_var=input_var)
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_in'],
                num_units=self.num_answers,
                nonlinearity=lasagne.nonlinearities.softmax)
        print "Done building final MLP ..."
        return net
        
    def build_model(self):
        qX, mask, iX, Y = self.qX, self.lstm_mask, self.iX, self.Y
        q_lstm_net = self.build_question_lstm(qX, mask)
        self.ql_out = lasagne.layers.get_output(q_lstm_net['l_dense'])
        mlp_input = self.combine_image_question_model(self.ql_out, iX)
        network = self.build_mlp_model(mlp_input)['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=True)
        loss = lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(network)
        self.inst_params = params
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_prediction = T.argmax(test_prediction, axis=1)
        print "Compiling..."
        train = theano.function([qX, mask, iX, Y], loss, updates=updates, allow_input_downcast=True)
        predict = theano.function([qX, mask, iX], test_prediction, allow_input_downcast=True)
        print "Done Compiling final model..."
        return train,predict

    def train(self):
        train, predict = self.build_model()
               
        
        







