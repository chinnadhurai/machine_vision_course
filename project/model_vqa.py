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

class vqa_type:
    def __init__(self, config):
        self.config = config
        self.X = T.fmatrix()
        self.Y = T.ivector()
        print "Initialized conv_classifier..."
    
    def build_lstm(self):
        MIN_LENGTH = 1
        MAX_LENGTH = 20
        # Number of units in the hidden (recurrent) layer
        N_HIDDEN = 100
        # Number of training sequences in each batch
        N_BATCH = 1000
        # Optimization learning rate
        LEARNING_RATE = .001
        # All gradients above this will be clipped
        GRAD_CLIP = 100
        # How often should we check the output?
        EPOCH_SIZE = 100
        # Number of epochs to train the net
        NUM_EPOCHS = 10
        
        l_in = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))
        l_mask = lasagne.layers.InputLayer(shape=(N_BATCH, MAX_LENGTH))

        l_forward = lasagne.layers.RecurrentLayer( \
        l_in, N_HIDDEN, mask_input=l_mask, grad_clipping=GRAD_CLIP, \
        W_in_to_hid=lasagne.init.HeUniform(), \
        W_hid_to_hid=lasagne.init.HeUniform(), \
        nonlinearity=lasagne.nonlinearities.tanh, only_return_final=True)
              
    
    
