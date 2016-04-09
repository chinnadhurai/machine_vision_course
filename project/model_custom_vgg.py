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

class vgg_feature:
    def __init__(self, config):
        self.config = config
        self.X_image = T.ftensor4()
        self.X = T.fmatrix()
        self.Y = T.ivector()
        print "Initialized conv_classifier..."
    
    def load_params(self):
        param_loc = self.config['vgg_params']
        params = l.load_params_pickle(param_loc)
        return params['param values']
        
    def compile_vgg_model(self):
        X = self.X_image
        params = self.load_params()
        network = vgg_16.build_model(X)
        self.net_vgg = network
        test_prediction = lasagne.layers.get_output(network['fc8'], deterministic=True)
        lasagne.layers.set_all_param_values(network['fc8'],params)
        self.predict_vgg = theano.function([X],test_prediction, allow_input_downcast=True)
        print "Done compiling vgg net model..."

    def create_vgg_feature_dataset(self, root_folder, output_folder):
        self.compile_vgg_model()
        for mode in ['val','test']:      
            files = [f for f in os.listdir(root_folder) if mode in str(f) and f.endswith('.npy')]
            for image_file in files:
                try:
                    im_id = re.findall(r'\d+', image_file)[-1]
                except IndexError :
                    continue
                feature_file = os.path.join( output_folder, mode + "_feature_" + str(im_id) )   
                self.create_dataset_util( os.path.join(root_folder,image_file), feature_file)
        print "Done creating dataset ..."

    def create_dataset_util(self, image_file, feature_file):
        print "Creating dataset from vgg net from...",image_file
        X = np.load(image_file)
        featureX = np.zeros((len(X), 1000))
        i = 0
        mbsize = 128
        total = len(range(0, len(X), mbsize))
        for start, end in zip(range(0, len(X), mbsize), range(mbsize, len(X), mbsize)):
            featureX[start:end] = self.predict_vgg(X[start:end])
            i += 1
            percent = (i*100)/total
            l.print_overwrite("Data percentage done %: ",percent)
        print "\nSaving data into", feature_file
        np.save(feature_file, featureX)

