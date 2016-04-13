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
import gzip
import nltk

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
        self.num_answers                = 1000
        self.bptt_trunk_steps           = -1
        self.grad_clip                  = 100
        self.batch_size                 = 128
        self.max_seq_length             = 10
        self.params                     = []
        pprint(config)
        print "\n----------------------"
        print "Initialization done ..."
    
    def add_to_param_list(self,l_params):
        for p in l_params:
            self.params.append(p)

    def load_params(self):
        param_loc = self.config['vgg_params']
        params = l.load_params_pickle(param_loc)
        return params['param values']
    
    def build_question_lstm(self, input_var, mask=None):
        input_dim, seq_len, mb_size = self.q_embed_dim, self.max_seq_length, self.batch_size
        # (batch size, max sequence length, number of features)
        l_in = lasagne.layers.InputLayer(shape=(None, seq_len, input_dim),
                                            input_var=input_var)
        l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask)
        l_lstm = lasagne.layers.LSTMLayer(l_in, 
                                          num_units             = self.q_embed_dim, 
                                          only_return_final     = True,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          grad_clipping         = self.grad_clip,
                                          mask_input            = l_mask
                                         )
        l_dense = lasagne.layers.DenseLayer(l_lstm, num_units=self.mlp_input_dim)
        self.add_to_param_list( lasagne.layers.get_all_params(l_dense) )
        net  = {'l_in':l_in, 'l_lstm':l_lstm, 'l_dense':l_dense}
        print "Done building question LSTM ..."
        return net
    
    def build_vgg_feature_mlp(self, input_var):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,1000),
                                         input_var=input_var)
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_in'],
                num_units=self.mlp_input_dim,
                nonlinearity=lasagne.nonlinearities.softmax)
        self.add_to_param_list( lasagne.layers.get_all_params(net['l_out']) )
        print "Done building vgg feature MLP ..."
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
        self.add_to_param_list( lasagne.layers.get_all_params(net['l_out']) )
        print "Done building final MLP ..."
        return net
        

    def build_model(self):
        if not self.config['fine_tune_vgg']:
            iX = self.iX
        else:
            self.X_image = T.ftensor4()
            params = self.load_params()
            network = vgg_16.build_model(self.X_image)
            self.net_vgg = network
            iX = lasagne.layers.get_output(network['fc8'], deterministic=True)
            lasagne.layers.set_all_param_values(network['fc8'],params)
            self.add_to_param_list(params)
        return self.build_model_util(iX)
    
    def build_model_util(self,iX):
        qX, mask, Y = self.qX, self.lstm_mask, self.Y
        q_lstm_net = self.build_question_lstm(qX, mask)
        ql_out = lasagne.layers.get_output(q_lstm_net['l_dense'])
        vgg_mlp_net = self.build_vgg_feature_mlp(iX)
        vgg_out = lasagne.layers.get_output(vgg_mlp_net['l_out'])
        mlp_input = self.combine_image_question_model(ql_out, vgg_out)
        network = self.build_mlp_model(mlp_input)['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=True)
        loss = lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.params#lasagne.layers.get_all_params(network)
        print len(params)
        print [ p.get_value().shape for p in params ]
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
        #self.train_util(qX, iX, Y, train, predict)
        
    def train_util(self, qX, iX, Y, train, predict):
        mb_size = self.batch_size
        for s,e in zip( range(0, len(qX), mb_size), range(mb_size, len(qX), mb_size)):
            mask = self.get_mask(qX[s:e])
            loss = train(qX[s:e], mask, iX[s:e], Y[s:e])
        cumsum = 0
        for s,e in zip( range(0, len(qX), mb_size), range(mb_size, len(qX), mb_size)):
            mask = self.get_mask(qX[s:e])
            pred = predict(qX[s:e], mask ,iX[s:e])
            cumsum += np.sum(pred == Y[s:e])
        print "Training accuracy(in  % )           :", cumsum*100 / Y.shape[0]       
        
    def get_input_question(self, image_ids, mode='train'):
        qfolder = os.path.join( self.config["dpath"],"real_images/questions")
        print "getting data from...", qfolder
        get_stored_vocab = True
        pfile = os.path.join(qfolder, "qvocab.zip")
        if not get_stored_vocab:
            self.qvocab, self.qword, self.max_qlen = load_data.get_question_vocab(qfolder)
            pickle.dump( [self.qvocab, self.qword, self.max_qlen], gzip.open( pfile, "wb" ) )            
        else:
            print "gettig data from pickle file", pfile
            self.qvocab, self.qword, self.max_qlen = pickle.load( gzip.open( pfile, "rb" ) )
        qdict = load_data.load_questions(qfolder, mode) 
        q_a = np.ones((len(image_ids),3,self.max_qlen),dtype='uint32' )*-1
        for itr, im_id in enumerate(image_ids):
            for i in range(3):
                q_id = im_id*10 + i
                q = qdict[q_id]['question']
                l_a = [ self.qvocab[w] for w in nltk.word_tokenize(str(q)) ]
                q_a[itr,i,:len(l_a)] = np.array(l_a, dtype='uint32')
        return q_a

    def get_question_util(self, file_id, mode='train'):
        ifile =  os.path.join( self.config["dpath"],"real_images/cleaned_images/" + str(mode).lower()+"_image.npy")
        image_ids = np.load(ifile)[file_id]
        print len(image_ids)
        q_a = self.get_input_question(image_ids,mode)
        print q_a.shape,q_a[np.random.randint(100),0,:]
        return q_a
                                  
    def get_answer_util(self,file_id):
        ifile =  os.path.join( self.config["dpath"],"real_images/cleaned_images/" + str(mode).lower()+"_image.npy")
        image_ids = np.load(ifile)[file_id]
        print len(image_ids)
        ans = self.get_ans(image_ids,mode)
        
