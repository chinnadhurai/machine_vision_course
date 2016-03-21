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
from lasagne.regularization import regularize_layer_params, l2
sys.dont_write_bytecode = True

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
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,1000),
                                         input_var=input_var)
        net['l_out'] = lasagne.layers.DenseLayer( 
                net['l_in'], 
                num_units=10,
                nonlinearity=lasagne.nonlinearities.softmax)
        return net

    def compile_logistic_model(self, lamda, input_params=None):
        X,Y = self.X,self.Y
        net = self.build_model(X)
        network = net['l_out']
        self.net_logistic = network
        prediction = lasagne.layers.get_output(network)
        loss = lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean() 
        for key in net.keys():
            loss += lamda*regularize_layer_params(net[key], l2)
        if input_params:
            print"Compiling classifier with input params..."
            lasagne.layers.set_all_param_values( net['l_out'],
                                                 [i.get_vlue() for i in input_params])
        params = lasagne.layers.get_all_params(network)
        self.inst_params = params
        updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.99)
        
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_prediction = T.argmax(test_prediction, axis=1)
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
        l.dump_h5(self.config['dataset_file'], [featuretrX,trY,featureteX,teY])

    def train(self):
        lamda_list = self.config["lamda_list"] 
        if self.config['load_dataset_file'] == False:
            self.create_dataset()
        trX, trY, teX, teY = l.load_h5(self.config['dataset_file'])
        slices = np.arange(trX.shape[0])
        np.random.shuffle(slices)
        val_size = trX.shape[0]/10
        vaX = trX[slices[:val_size]]
        vaY = trY[slices[:val_size]]
        #trX = trX[slices[val_size:]]
        #trY = trY[slices[val_size:]]
        print "Training size      :", trX.shape[0]
        print "Validation size    :", vaX.shape[0] 
        print "Test size          :", teX.shape[0]
        mbsize = self.config['mini_batch_size']
        max_val = -1
        for lamda in lamda_list:
            print "\nLamda :", lamda
            max_val_per_lamda = -1
            train_logistic,predict_logistic = self.compile_logistic_model(lamda)
            for i in range(self.config['epochs']):
                #print "epoch :",i
                for start, end in zip(range(0, len(trX), mbsize), range(mbsize, len(trX), mbsize)):
                    cost = train_logistic(trX[start:end], trY[start:end])
                    #l.print_overwrite("cost : ",cost)
                trM =  np.mean( trY[:50000] == predict_logistic(trX[:50000]))
                vaM =  np.mean( vaY[:10000] == predict_logistic(vaX[:10000]))
                l.print_overwrite("validation accuracy % : ", vaM*100)
                if ( max_val_per_lamda <= vaM ):
                    max_val_per_lamda = vaM
                    self.max_params_per_lamda = self.inst_params
                    #print "\n  train accracy % :", trM*100 ,"  validation accuracy %: ",teM*100
            print "\nMAX validation accuracy % : ", max_val_per_lamda*100    
            if (max_val < max_val_per_lamda):
                max_val = max_val_per_lamda
                self.lamda = lamda
                self.best_params = self.max_params_per_lamda
            
        train_logistic,predict_logistic = self.compile_logistic_model(lamda, self.best_params)
        teM =  np.mean( teY[:10000] == predict_logistic(teX[:10000]))
        print "best lamda   :", self.lamda
        print "Test Accuracy:", teM*100




