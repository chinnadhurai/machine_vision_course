_author__ = 'chinna'
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
import json
import time
import datetime
from nltk.tokenize import WordPunctTokenizer

class MODEL:
    def __init__( self,
                  config,
                  qvocab_len,
                  max_qlen,
                  num_ans,
                  num_qtypes,  
                  l_saver):
        self.config                     = config
        self.qn                         = T.imatrix()
        self.lstm_mask                  = T.imatrix()
        self.iX                         = T.fmatrix()
        self.Y                          = T.ivector()
        self.qtype                      = T.ivector()
        self.sparse_indices             = T.ivector()
        self.qembd                      = T.fmatrix()
        self.ql_out                     = T.fmatrix()
        self.timer                      = l.timer_type()
        self.saver, self.exp_saver      = l_saver
        self.qlstm_hidden_dim           = 300 
        self.qn_classifier_emb_size     = 75
        self.max_ql                     = max_qlen
        self.qvocab_len                 = qvocab_len 
        self.bptt_trunk_steps           = -1 
        self.mlp_input_dim              = 1024
        self.num_qtypes                 = num_qtypes
        self.num_ans                    = num_ans
        self.grad_clip                  = config['grad_clip']
        self.params                     = {}
        print "Models Initialization done ..."
        
    def add_to_param_list(self,network,l_params,param_type):
        print "Adding params",param_type
        self.params[param_type] = l_params
        self.load_saved_params(network,param_type)

    def get_params(self,l_param_types):
        params = []
        for mp in l_param_types:
            for p in self.params[mp]:
                params.append(p)
        return params    
    
    def dump_current_params(self):
        params = []
        for k,v in self.params.items():
            params = [p.get_value() for p in v]
            self.exp_saver.save_array(params,fid=str(k) + '_model_params')

    def load_saved_params(self, network, param_type):
        if not self.config['load_from_saved_params']:
            return
        print "Loading saved params for model :", param_type
        params = self.exp_saver.load_array(fid=str(param_type) + '_model_params')
        lasagne.layers.set_all_param_values(network, params)

    def add_params(self, network, param_type):
        self.add_to_param_list( network, lasagne.layers.get_all_params(network) , param_type)
    
    def question_boW(self, input_var, param_type=None):
        input_dim, seq_len = self.qvocab_len, self.max_ql

        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),
                                         input_var=input_var)

        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_size=input_dim, output_size=self.qlstm_hidden_dim)
        net  = {'l_in':l_in, 'l_out':l_embd}
        if param_type is not None:
            self.add_params(l_embd, param_type)
        print "Done building question_boW ..."
        return net

    def single_layer_question_lstm(self, l_input_var, param_type=None):
        input_dim, seq_len = self.qvocab_len, self.max_ql
        input_var, mask = l_input_var
        # (batch size, max sequence length, number of features)
     
        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),
                                            input_var=input_var)
        lstm_params = lasagne.layers.get_all_params(l_in)
        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_dim, self.qlstm_hidden_dim)
        l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask)
        l_lstm = lasagne.layers.LSTMLayer(l_embd, 
                                          num_units             = self.qlstm_hidden_dim, 
                                          only_return_final     = True,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          mask_input            = l_mask
                                         )
        l_dense = lasagne.layers.DenseLayer(l_lstm, num_units=self.mlp_input_dim)
        net  = {'l_in':l_in, 'l_lstm':l_lstm, 'l_out':l_dense}
        if param_type is not None:
            self.add_params(l_dense, param_type)
        print "Done building single_layer_question_lstm..."
        return net
 
    def double_layer_question_lstm(self, l_input_var, param_type=None):
        input_dim, seq_len = self.qvocab_len, self.max_ql
        input_var, mask = l_input_var
        # (batch size, max sequence length, number of features)

        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),
                                            input_var=input_var)
        lstm_params = lasagne.layers.get_all_params(l_in)
        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_dim, self.qlstm_hidden_dim)
        l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask)
        l_lstm1 = lasagne.layers.LSTMLayer(l_embd,
                                          num_units             = self.qlstm_hidden_dim,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          mask_input            = l_mask
                                         )
        l_lstm2 = lasagne.layers.LSTMLayer(l_lstm1,
                                          num_units             = self.qlstm_hidden_dim,
                                          only_return_final     = True,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          mask_input            = l_mask
                                         )
        l_dense = lasagne.layers.DenseLayer(l_lstm2, num_units=self.mlp_input_dim)
        net  = {'l_in':l_in, 'l_lstm2':l_lstm2, 'l_out':l_dense}
        if param_type is not None:
            self.add_params(l_dense, param_type)
        print "Done building double_layer_question_lstm..."
        return net
   
    def bidirectional_question_lstm(self, l_input_var, param_type=None):
        input_dim, seq_len = self.qvocab_len, self.max_ql
        input_var, mask = l_input_var
        # (batch size, max sequence length, number of features)

        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),
                                            input_var=input_var)
        lstm_params = lasagne.layers.get_all_params(l_in)
        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_dim, self.qlstm_hidden_dim)
        l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask)
        l_lstmf = lasagne.layers.LSTMLayer(l_embd,
                                          num_units             = self.qlstm_hidden_dim,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          mask_input            = l_mask,
                                          only_return_final     = True
                                          )
        l_lstmb = lasagne.layers.LSTMLayer(l_embd,
                                          num_units             = self.qlstm_hidden_dim,
                                          only_return_final     = True,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          mask_input            = l_mask,
                                          backwards             = True
                                          )
        l_concat = lasagne.layers.ConcatLayer([l_lstmf, l_lstmb])
        l_dense = lasagne.layers.DenseLayer(l_concat, num_units=self.mlp_input_dim-self.qn_classifier_emb_size)
        net  = {'l_in':l_in, 'l_lstm2':l_lstmb, 'l_out':l_dense}
        if param_type is not None:
            self.add_params(l_dense, param_type)
        print "Done building bidirectional_question_lstm..."
        return net




    def qn_classifier_lstm(self, l_input_var, param_type=None):
        input_dim, seq_len = self.qvocab_len, self.max_ql
        # (batch size, max sequence length, number of features)
        input_var,mask = l_input_var
        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),
                                            input_var=input_var)
        lstm_params = lasagne.layers.get_all_params(l_in)
        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_dim, self.qn_classifier_emb_size)
        l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask)
        l_lstm = lasagne.layers.LSTMLayer(l_embd,
                                          num_units             = self.qn_classifier_emb_size,
                                          only_return_final     = True,
                                          gradient_steps        = self.bptt_trunk_steps,
                                          mask_input            = l_mask
                                         )
        l_dense = lasagne.layers.DenseLayer(l_lstm, num_units=self.qn_classifier_emb_size)
        net  = {'l_in':l_in, 'l_lstm':l_lstm, 'l_out':l_dense}
        if param_type is not None:
            self.add_params(l_dense, param_type)
        print "Done building qn_classifier_lstm ..."
        return net    

    def qn_classifier_mlp(self,input_var,param_type=None):
        num_qn_types, input_dim = self.num_qtypes,self.qn_classifier_emb_size
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,input_dim),
                                         input_var=input_var)
        
        net['l_h1'] =  lasagne.layers.DenseLayer( net['l_in'],
                                                  num_units=1000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        net['l_h1_drop'] = lasagne.layers.DropoutLayer(net['l_h1'], p=0.5)

        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_h1_drop'],
                num_units=num_qn_types,
                nonlinearity=lasagne.nonlinearities.softmax)
        if param_type is not None:
            self.add_params(net['l_out'], param_type)
        print "Done building qn classifier MLP ..."
        return net

    def qmbd_mlp_model(self,input_var, param_type=None):    
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,self.qn_classifier_emb_size),
                                         input_var=input_var)

        net['l_h1'] =  lasagne.layers.DenseLayer( net['l_in'],
                                                  num_units=1000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        net['l_h1_drop'] = lasagne.layers.DropoutLayer(net['l_h1'], p=0.5)

        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_h1_drop'],
                num_units=self.mlp_input_dim,
                nonlinearity=lasagne.nonlinearities.softmax)
        if param_type is not None:
            self.add_params(net['l_out'], param_type)
        print "Done building question embd mlp ..."
        return net

    def vgg_feature_mlp(self, input_var,param_type=None):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,1000),
                                         input_var=input_var)
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_in'],
                num_units=self.mlp_input_dim,
                nonlinearity=lasagne.nonlinearities.rectify)
        if param_type is not None:
            self.add_params(net['l_out'], param_type)
        print "Done building vgg feature MLP ..."
        return net

    def final_mlp_model(self,input_var,param_type=None):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,2*self.mlp_input_dim),
                                         input_var=input_var)

        net['l_h1'] =  lasagne.layers.DenseLayer( net['l_in'],
                                                  num_units=1000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        net['l_h1_drop'] = lasagne.layers.DropoutLayer(net['l_h1'], p=0.5)
        
        net['l_h2'] =  lasagne.layers.DenseLayer( net['l_h1_drop'],
                                                  num_units=1000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)
        
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_h2'],
                num_units=self.num_ans ,
                nonlinearity=lasagne.nonlinearities.softmax)
        if param_type is not None:
            self.add_params(net['l_out'], param_type)
        print "Done building final MLP ..."
        return net


    def st_qmbd_mlp_model(self,input_var, param_type=None):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,2400),
                                         input_var=input_var)

        net['l_h1'] =  lasagne.layers.DenseLayer( net['l_in'],
                                                  num_units=2000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        #net['l_h1_drop'] = lasagne.layers.DropoutLayer(net['l_h1'], p=0.5)


        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_h1'],
                num_units=self.mlp_input_dim)
                #nonlinearity=lasagne.nonlinearities.softmax)
        if param_type is not None:
            self.add_params(net['l_out'], param_type)
        print "Done building skip_thought to question embd mlp ..."
        return net

    def conv_final_mlp_model(self,input_var,param_type=None):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,1,2*self.mlp_input_dim),
                                         input_var=input_var)
        
        net['l_conv'] = lasagne.layers.Conv1DLayer( net['l_in'], 
                                                    pad='same',
                                                    num_filters=10, 
                                                    filter_size=50 )
    
        net['l_conv'] = lasagne.layers.Conv1DLayer( net['l_conv'],
                                                    num_filters=5, 
                                                    filter_size=30 )
        
        net['l_conv'] = lasagne.layers.MaxPool1DLayer(net['l_conv'], pool_size=2)
        
        net['l_h1'] =  lasagne.layers.DenseLayer( net['l_conv'],
                                                  num_units=2000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        net['l_h1'] = lasagne.layers.DropoutLayer(net['l_h1'], p=0.5)

        net['l_h2'] =  lasagne.layers.DenseLayer( net['l_h1'],
                                                  num_units=1000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)
        
        net['l_h2'] = lasagne.layers.DropoutLayer(net['l_h2'], p=0.5)
        
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_h2'],
                num_units=self.num_ans ,
                nonlinearity=lasagne.nonlinearities.softmax)


        if param_type is not None:
            self.add_params(net['l_out'], param_type)
        print "Done building final MLP ..."
        return net

    #---------------------------------------------------
    #
    #  COMPOSED MODELS
    #
    #---------------------------------------------------


    # build_vqa_model1
    def build_vqa_model_vanilla(self):
        qn, mask, Y = self.qn, self.lstm_mask, self.Y
        sparse_indices, qembd, iX = self.sparse_indices, self.ql_out, self.iX
        l_param_type = [ 'vgg_feature_mlp', 'qlstm', 'final_mlp' ]

        vgg_mlp_net = self.vgg_feature_mlp(iX,param_type=l_param_type[0])['l_out']
        vgg_out = lasagne.layers.get_output(vgg_mlp_net)

        ql_out = self.double_layer_question_lstm([qn, mask], param_type=l_param_type[1])['l_out']
        ql_out = lasagne.layers.get_output(ql_out)

        mlp_input = ql_out * vgg_out
        network = self.final_mlp_model(mlp_input, param_type=l_param_type[2])['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=False)
        
        loss =  lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.get_params(l_param_type)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)#sparse_indices=sparse_indices, deterministic=True)
        test_prediction = T.argmax(test_prediction, axis=1)
        acc = T.mean(T.eq(test_prediction, Y),dtype=theano.config.floatX)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qn, mask, iX, Y], [loss], updates=updates, allow_input_downcast=True)
        ans_predict = theano.function([qn, mask, iX, Y], acc, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling final model..."
        return train, ans_predict
    

    


    # sparse_ids
    def build_vqa_model_sparse_ids(self):
        qn, mask, Y = self.qn, self.lstm_mask, self.Y
        sparse_indices, qembd, iX = self.sparse_indices, self.ql_out, self.iX
        l_param_type = [ 'sparse_vgg_feature_mlp', 'sparse_lstm', 'sparse_final_mlp' ]

        vgg_mlp_net = self.vgg_feature_mlp(iX,param_type=l_param_type[0])['l_out']
        vgg_out = lasagne.layers.get_output(vgg_mlp_net)

        ql_out = self.double_layer_question_lstm([qn, mask], param_type=l_param_type[1])['l_out']
        ql_out = lasagne.layers.get_output(ql_out)

        mlp_input = ql_out * vgg_out
        network = self.final_mlp_model(mlp_input, param_type=l_param_type[2])['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=False, sparse_indices=sparse_indices)
        prediction = T.nnet.softmax(prediction[:,sparse_indices])

        loss =  lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.get_params(l_param_type)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)

        test_prediction = lasagne.layers.get_output(network, sparse_indices=sparse_indices, deterministic=True)
        test_prediction = T.nnet.softmax(test_prediction[:,sparse_indices])
        test_prediction = T.argmax(test_prediction, axis=1)
        acc = T.mean(T.eq(test_prediction, Y),dtype=theano.config.floatX)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qn, mask, iX, Y, sparse_indices], [loss], updates=updates, allow_input_downcast=True)
        ans_predict = theano.function([qn, mask, iX, Y, sparse_indices], acc, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling final model..."
        return train, ans_predict






    # sparse_ids
    def build_vqa_model_sparse_ids_only_val(self):
        qn, mask, Y = self.qn, self.lstm_mask, self.Y
        sparse_indices, qembd, iX = self.sparse_indices, self.ql_out, self.iX
        l_param_type = [ 'sparse_vgg_feature_mlp', 'sparse_lstm', 'sparse_final_mlp' ]

        iX = iX / T.sqrt(iX).sum(axis=1).reshape((iX.shape[0], 1))
        vgg_mlp_net = self.vgg_feature_mlp(iX,param_type=l_param_type[0])['l_out']
        vgg_out = lasagne.layers.get_output(vgg_mlp_net)

        ql_out = self.single_layer_question_lstm([qn, mask], param_type=l_param_type[1])['l_out']
        ql_out = lasagne.layers.get_output(ql_out)

        mlp_input = ql_out * vgg_out
        network = self.final_mlp_model(mlp_input, param_type=l_param_type[2])['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=False)

        loss =  lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.get_params(l_param_type)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)

        test_prediction = lasagne.layers.get_output(network, sparse_indices=sparse_indices, deterministic=True)
        test_prediction = T.nnet.softmax(test_prediction[:,sparse_indices])
        test_prediction = T.argmax(test_prediction, axis=1)
        acc = T.mean(T.eq(test_prediction, Y),dtype=theano.config.floatX)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qn, mask, iX, Y], [loss], updates=updates, allow_input_downcast=True)
        ans_predict = theano.function([qn, mask, iX, Y, sparse_indices], acc, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling final model..."
        return train, ans_predict



    # sparse_ids
    def build_vqa_model_concat_class_embd(self):
        qn, mask, Y = self.qn, self.lstm_mask, self.Y
        sparse_indices, qembd, iX = self.sparse_indices, self.ql_out, self.iX
        l_param_type = [ 'sparse_vgg_feature_mlp', 'bi_dir_lstm', 'sparse_final_mlp' ]

        iX = iX / T.sqrt(iX).sum(axis=1).reshape((iX.shape[0], 1))
        vgg_mlp_net = self.vgg_feature_mlp(iX,param_type=l_param_type[0])['l_out']
        vgg_out = lasagne.layers.get_output(vgg_mlp_net)

        ql_out = self.bidirectional_question_lstm([qn, mask], param_type=l_param_type[1])['l_out']
        ql_out = lasagne.layers.get_output(ql_out)
        ql_out = T.concatenate([ql_out, qembd],axis=1)
        mlp_input = ql_out * vgg_out
        network = self.final_mlp_model(mlp_input, param_type=l_param_type[2])['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=False)

        loss =  lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.get_params(l_param_type)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_prediction = T.argmax(test_prediction, axis=1)
        acc = T.mean(T.eq(test_prediction, Y),dtype=theano.config.floatX)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qn, mask, iX, Y, qembd], [loss], updates=updates, allow_input_downcast=True)
        ans_predict = theano.function([qn, mask, iX, Y, qembd], acc, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling final model..."
        return train, ans_predict


    def build_vqa_model_skip_thought_vanilla(self):
        qembd, iX, Y = self.ql_out, self.iX, self.Y
        l_param_type = [ 'vgg_feature_mlp', 'skip_thought_mlp', 'final_mlp' ]
        
        vgg_mlp_net = self.vgg_feature_mlp(iX,param_type=l_param_type[0])['l_out']
        vgg_out = lasagne.layers.get_output(vgg_mlp_net)

        ql_out = self.st_qmbd_mlp_model(qembd,param_type=l_param_type[1] )['l_out']
        ql_out = lasagne.layers.get_output(ql_out)
        mlp_input = T.concatenate([ql_out,vgg_out],axis=1)
        network = self.final_mlp_model(mlp_input, param_type=l_param_type[2])['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=False)

        loss =  lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.get_params(l_param_type)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)#sparse_indices=sparse_indices, deterministic=True)
        test_prediction = T.argmax(test_prediction, axis=1)
        acc = T.mean(T.eq(test_prediction, Y),dtype=theano.config.floatX)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qembd, iX, Y], [loss], updates=updates, allow_input_downcast=True)
        ans_predict = theano.function([qembd, iX, Y], acc, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling vqa_model_skip_thought_vanilla model..."
        return train, ans_predict


    def build_vqa_model_skip_thought_conv(self):
        qembd, iX, Y = self.ql_out, self.iX, self.Y
        l_param_type = [ 'vgg_feature_mlp', 'skip_thought_mlp', 'final_conv_mlp' ]

        vgg_mlp_net = self.vgg_feature_mlp(iX,param_type=l_param_type[0])['l_out']
        vgg_out = lasagne.layers.get_output(vgg_mlp_net)

        ql_out = self.st_qmbd_mlp_model(qembd,param_type=l_param_type[1] )['l_out']
        ql_out = lasagne.layers.get_output(ql_out)

        mlp_input = T.concatenate([ql_out,vgg_out],axis=1)
        mlp_input = mlp_input.dimshuffle([0, 'x', 1])
        network = self.conv_final_mlp_model(mlp_input, param_type=l_param_type[2])['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=False)

        loss =  lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.get_params(l_param_type)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)

        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_prediction = T.argmax(test_prediction, axis=1)
        acc = T.mean(T.eq(test_prediction, Y),dtype=theano.config.floatX)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qembd, iX, Y], [loss], updates=updates, allow_input_downcast=True)
        ans_predict = theano.function([qembd, iX, Y], acc, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling vqa_model_skip_thought_vanilla model..."
        return train, ans_predict




    # Question classifier model
    def build_qn_type_model(self):
        qtype,qembd = self.qtype,self.qembd
        qn, mask =  self.qn, self.lstm_mask
        l_param_type = [ 'qn_class_lstm', 'qn_class_mlp' ]

        q_lstm_net = self.qn_classifier_lstm([qn, mask], param_type=l_param_type[0])['l_out']
        qembd = lasagne.layers.get_output(q_lstm_net)
        l2_penalty_qlstm = regularize_layer_params(q_lstm_net, l2)
        
        q_type_net = self.qn_classifier_mlp(qembd, param_type=l_param_type[1])['l_out']
        q_type_pred = lasagne.layers.get_output(q_type_net,deterministic=False)
        l2_penalty_mlp = regularize_layer_params(q_type_net, l2)

        loss = lasagne.objectives.categorical_crossentropy(q_type_pred, qtype)
        loss = loss.mean() + l2_penalty_mlp + l2_penalty_qlstm
        params = self.get_params(l_param_type)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]

        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.003)
        qtype_test_pred = lasagne.layers.get_output(q_type_net,deterministic=True)
        qtype_test_pred = T.argmax(qtype_test_pred, axis=1)
        acc = T.mean(T.eq(qtype_test_pred, qtype),dtype=theano.config.floatX)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qn,mask, qtype], loss, updates=updates, allow_input_downcast=True)
        qtype_predict = theano.function([qn,mask,qtype], [qtype_test_pred, acc], allow_input_downcast=True)
        qembd_fn = theano.function([qn,mask], qembd, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling qtype model..."
        return train, qtype_predict, qembd_fn




