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
from load_data import load_cifar_10_data_upsampled
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
        self.X_image = T.ftensor4()
        self.X = T.fmatrix()
        self.Y = T.ivector()
        print "Initialized conv_classifier..."

    def load_params(self):
        param_loc = self.config['params']
        params = l.load_params_pickle(param_loc)
        return params['param values']
    
    def build_model(self, input_var=None):
        l_in = lasagne.layers.InputLayer(shape=(None,1000),
                                         input_var=input_var)
        l_out = lasagne.layers.DenseLayer( 
                l_in, 
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)
        return l_out

    def compile_logistic_model(self):
        X,Y = self.X,self.Y
        network = self.build_model(X)
        self.net_logistic = network
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = lasagne.layers.get_all_params(network, trainable=True)
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
        
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,Y)
        train = theano.function([X, Y], loss, updates=updates, allow_input_downcast=True)      
        predict = theano.function([X], test_prediction, allow_input_downcast=True)
        print "Done Compiling logistic model..."
        return train,predict

    def compile_vgg_model(self):
        X = self.X_image
        params = self.load_params()
        network = vgg_16.build_model(X)
        self.net_vgg = network
        test_prediction = lasagne.layers.get_output(network['fc8'], deterministic=True)
        lasagne.layers.set_all_param_values(network['fc8'],params) 
        predict = theano.function([X],test_prediction, allow_input_downcast=True)
        print "Done compiling vgg net model..."
        return predict        

    def create_dataset(self):
        print "Creating dataset from vgg net..."
        predict_vgg = self.compile_vgg_model()
        trX, trY, teX, teY = load_cifar_10_data_upsampled(self.config)
        mbsize = self.config['mini_batch_size']
        featuretrX = np.zeros((len(trX), 1000))
        featureteX = np.zeros((len(teX), 1000))
        i = 0
        total = len(range(0, len(trX), mbsize))
        for start, end in zip(range(0, len(trX), mbsize), range(mbsize, len(trX), mbsize)):
            featuretrX[start:end] = predict_vgg(trX[start:end])
            i += 1
            percent = (i*100)/total
            l.print_overwrite("Traning data percentage done %: ",percent)
        i = 0
        total = len(range(0, len(teX), mbsize))
        for start, end in zip(range(0, len(teX), mbsize), range(mbsize, len(teX), mbsize)):
            featureteX[start:end] = predict_vgg(teX[start:end])
            i += 1
            percent = (i*100)/total
            l.print_overwrite("Test data percentage done %: ",percent)
        print "\nloading data into", self.config['dataset_file'] 
        l.dump_params_pickle(self.config['dataset_file'], [featuretrX,trY,featureteX,teY])

    def train(self):
        train_logistic,predict_logistic = self.compile_logistic_model()
        if self.config['load_dataset_file'] == False:
            self.create_dataset()
        trX, teY, teX, teY = l.load_params_pickle_gzip(self.config['dataset_file'])
        mbsize = self.config['mini_batch_size']
        for i in range(self.config['epochs']):
            print "epoch :",i
            for start, end in zip(range(0, len(trX), mbsize), range(mbsize, len(trX), mbsize)):
                cost = train_logistic(trX[start:end], trY[start:end])
                l.print_overwrite("cost : ",cost)
            trM =  np.mean( trY[:20000] == predict_logistic(trX[:20000]))
            teM =  np.mean( teY[:10000] == predict_logistic(teX[:10000]))
            print "  train accracy :", trM ,"  validation accuracy : ",teM
