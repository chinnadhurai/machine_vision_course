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

class vqa_type:
    def __init__(self, config):
        print "Configuration :"
        self.config                     = config
        self.qX                         = T.imatrix()#T.ftensor3()
        self.lstm_mask                  = T.imatrix()
        self.iX                         = T.fmatrix()
        self.Y                          = T.ivector()
        self.qtype                      = T.ivector()
        self.sparse_indices             = T.ivector()
        self.qembd                      = T.fmatrix()    
        self.params                     = []
        self.ans_type_dict              = {'other': 1, 'yes/no': 0, 'number': 2}
        pprint(config)
        print "\n----------------------"
        print "\nPreping data set..."
        self.timer = l.timer_type()
        self.saver = l.save_np_arrays(os.path.join(self.config['questions_folder'], "temp"))
        self.exp_saver = l.save_np_arrays(os.path.join(self.config['real_abstract_images'], config['experiment_id']))
        default_plot_folder = os.path.join(config['real_abstract_images'], "plots")
        self.plotter = l.plotter_tool(os.path.join(default_plot_folder, config['experiment_id']))
        self.tokenizer = WordPunctTokenizer()
        pfile = os.path.join(self.config['questions_folder'], "qn_vocab.zip")
        self.qvocab, self.qword, self.max_qlen = pickle.load( gzip.open( pfile, "rb" ) )
        pfile = os.path.join(self.config['annotations_folder'], "top1000_ans_vocab.zip")
        self.avocab, self.aword = pickle.load( gzip.open( pfile, "rb" ) )
        print "Answer vocab size    :", len(self.avocab)
        print "question vocab size  :", len(self.qvocab)
        pfile = os.path.join(self.config['annotations_folder'], "id_info.zip")
        self.id_info = pickle.load( gzip.open( pfile, "rb" ) )
        self.num_division = config['num_division']
        self.grad_clip = config['grad_clip']
        self.qdict, self.image_ids, self.question_ids, self.answer_types = {},{},{},{}
        self.answers, self.questions, self.mask, self.divisions, self.saved_params = {},{},{},{},{}
        self.timer.set_checkpoint('init')
        load_from_file= True
        for mode in ['train','val']:
            self.qdict[mode] = load_data.load_questions(self.config['questions_folder'], mode=mode)
            self.image_ids[mode], self.question_ids[mode], self.answer_types[mode], self.answers[mode] = self.get_image_question_ans_ids(mode, load_from_file=load_from_file)
            mbsize = len(self.image_ids[mode]) // self.num_division
            self.divisions[mode] = zip(range(0, len(self.image_ids[mode]), mbsize), range(mbsize, len(self.image_ids[mode]), mbsize))
            self.store_question_data(mode, load_from_file=load_from_file, save_image_features=False)
            self.answer_type_util(mode=mode)
        print "Init time taken(in mins)", self.timer.print_checkpoint('init')
        self.p_ans_vector = self.unigram_dist()
        print "Initialization done ..."
        
    def add_to_param_list(self,network,l_params,param_type):
        for p in l_params:
            self.params.append(p)
        local_dict = {}
        local_dict['params'] = l_params
        self.saved_params[param_type] = local_dict
        self.load_saved_params(network,param_type)
    
    def dump_current_params(self):
        #print "Saving current params to ", self.config['saved_params']
        params = []
        for k,v in self.saved_params.items():
            params = [p.get_value() for p in v['params']]
            self.exp_saver.save_array(params,fid=str(k) + '_model_params')

    def load_saved_params(self, network, param_type):
        if not self.config['load_from_saved_params']:
            return
        print "Loading saved params for model :", param_type
        params = self.exp_saver.load_array(fid=str(param_type) + '_model_params')
        lasagne.layers.set_all_param_values(network, params)

    def load_vgg_params(self):
        param_loc = self.config['vgg_params']
        params = l.load_params_pickle(param_loc)
        return params['param values']
    
    def build_question_boW(self, input_var, mask=None):
        input_dim, seq_len = len(self.qvocab), 6#self.max_qlen

        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),
                                         input_var=input_var)

        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_size=len(self.qvocab), output_size=self.config['lstm_hidden_dim'])
        net  = {'l_in':l_in, 'l_embd':l_embd}
        print "Done building question LSTM ..."
        return net
    

    def build_1layer_question_lstm(self, input_var, mask=None):
        input_dim, seq_len = len(self.qvocab), self.max_qlen
        # (batch size, max sequence length, number of features)
     
        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),#, input_dim),
                                            input_var=input_var)
        lstm_params = lasagne.layers.get_all_params(l_in)
        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_dim, self.config['lstm_hidden_dim'])
        l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask)
        l_lstm = lasagne.layers.LSTMLayer(l_embd, 
                                          num_units             = self.config['lstm_hidden_dim'], 
                                          only_return_final     = True,
                                          gradient_steps        = self.config['bptt_trunk_steps'],
                                          mask_input            = l_mask
                                         )
        l_dense = lasagne.layers.DenseLayer(l_lstm, num_units=self.config['mlp_input_dim'])
        self.add_to_param_list( l_dense, lasagne.layers.get_all_params(l_dense) , param_type='qlstm')
        net  = {'l_in':l_in, 'l_lstm':l_lstm, 'l_dense':l_dense}
        print "Done building question LSTM ..."
        return net
    
    def build_qn_classifier_lstm(self, input_var, mask=None):
        input_dim, seq_len = len(self.qvocab), self.max_qlen
        # (batch size, max sequence length, number of features)

        l_in = lasagne.layers.InputLayer(shape=(None, seq_len),#, input_dim),
                                            input_var=input_var)
        lstm_params = lasagne.layers.get_all_params(l_in)
        l_embd = lasagne.layers.EmbeddingLayer(l_in, input_dim, 75)#self.config['lstm_hidden_dim'])
        l_mask = lasagne.layers.InputLayer(shape=(None, seq_len), input_var=mask)
        l_lstm = lasagne.layers.LSTMLayer(l_embd,
                                          num_units             = 75,#self.config['lstm_hidden_dim'],
                                          only_return_final     = True,
                                          gradient_steps        = self.config['bptt_trunk_steps'],
                                          mask_input            = l_mask
                                         )
        l_dense = lasagne.layers.DenseLayer(l_lstm, num_units=75)#self.config['mlp_input_dim'])
        self.add_to_param_list( l_dense, lasagne.layers.get_all_params(l_dense) , param_type='qlstm')
        net  = {'l_in':l_in, 'l_lstm':l_lstm, 'l_dense':l_dense}
        print "Done building question LSTM ..."
        return net    

    def build_qn_classifier_mlp(self,input_var):
        num_qn_types, input_dim = len(self.ans_type_dict),75#1024# self.config['lstm_hidden_dim']
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
        #self.add_to_param_list( net['l_out'], lasagne.layers.get_all_params(net['l_out']), param_type='qn_classifier' )
        print "Done building qn classifier MLP ..."
        return net

    def build_vgg_feature_mlp(self, input_var):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,1000),
                                         input_var=input_var)
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_in'],
                num_units=self.config['mlp_input_dim'],
                nonlinearity=lasagne.nonlinearities.rectify)
        self.add_to_param_list( net['l_out'], lasagne.layers.get_all_params(net['l_out']), param_type='vgg2mlp' )
        print "Done building vgg feature MLP ..."
        return net

    def combine_image_question_model(self,image_feature, question_feature):
        return image_feature*question_feature#T.concatenate([image_feature, question_feature],axis=1 )
    
    def build_mlp_model(self,input_var):
        net = {}
        net['l_in'] = lasagne.layers.InputLayer(shape=(None,self.config['mlp_input_dim']),
                                         input_var=input_var)

        net['l_h1'] =  lasagne.layers.DenseLayer( net['l_in'],
                                                  num_units=1000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)

        net['l_h1_drop'] = lasagne.layers.DropoutLayer(net['l_h1'], p=0.5)
        
        net['l_h2'] =  lasagne.layers.DenseLayer( net['l_h1_drop'],
                                                  num_units=1000,
                                                  nonlinearity=lasagne.nonlinearities.rectify)
        
        #net['l_out'] = lasagne.layers.DropoutLayer(net['l_h2'], p=0.5)
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_h2'],
                num_units=len(self.avocab),
                nonlinearity=lasagne.nonlinearities.softmax)
        self.add_to_param_list( net['l_out'], lasagne.layers.get_all_params(net['l_out']), param_type='final_mlp' )
        print "Done building final MLP ..."
        return net
    

    def build_model(self):
        if not self.config['fine_tune_vgg']:
            iX = self.iX
        else:
            self.X_image = T.ftensor4()
            params = self.load_vgg_params()
            network = vgg_16.build_model(self.X_image)
            self.net_vgg = network
            iX = lasagne.layers.get_output(network['fc8'], deterministic=True)
            lasagne.layers.set_all_param_values(network['fc8'],params)
            self.add_to_param_list(network, lasagne.layers.get_all_param_values(network['fc8']), param_type='vgg16')
        return self.build_model_util(iX)
    
    def build_model_util(self,iX):
        qX, mask, Y, sparse_indices = self.qX, self.lstm_mask, self.Y, self.sparse_indices
        q_lstm_net = self.build_1layer_question_lstm(qX, mask)
        ql_out = lasagne.layers.get_output(q_lstm_net['l_dense'])
        vgg_mlp_net = self.build_vgg_feature_mlp(iX)
        vgg_out = lasagne.layers.get_output(vgg_mlp_net['l_out'])
        mlp_input = self.combine_image_question_model(ql_out, vgg_out)
        network = self.build_mlp_model(mlp_input)['l_out']
        prediction = lasagne.layers.get_output(network, sparse_indices=sparse_indices, deterministic=False)
        prediction = T.nnet.softmax(prediction[:,sparse_indices])
        loss =  lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.params
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)
        
        test_prediction = lasagne.layers.get_output(network, sparse_indices=sparse_indices, deterministic=True)
        test_prediction = T.nnet.softmax(test_prediction[:,sparse_indices])
        test_prediction = T.argmax(test_prediction, axis=1)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qX, mask, iX, Y, sparse_indices], [loss], updates=updates, allow_input_downcast=True)
        ans_predict = theano.function([qX, mask, iX, sparse_indices], test_prediction, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling final model..."
        return train, ans_predict

    def build_qn_type_model(self, from_scratch=False):
        qtype,qembd = self.qtype,self.qembd
        qX, mask =  self.qX, self.lstm_mask
        if from_scratch:
            #q_bow_net = self.build_question_boW(qX)
            #q_bow = lasagne.layers.get_output(q_bow_net['l_embd'])
            #l2_penalty_qbow = regularize_layer_params(q_bow_net['l_embd'], l2)
            #qbow_params = lasagne.layers.get_all_params(q_bow_net['l_embd'])
            #qembd = T.sum(q_bow,axis=1)
            q_lstm_net = self.build_qn_classifier_lstm(qX, mask)
            qlstm_params = lasagne.layers.get_all_params(q_lstm_net['l_dense'])
            l2_penalty_qlstm = regularize_layer_params(q_lstm_net['l_dense'], l2)
            #l2_penalty_qlstm += regularize_layer_params(q_lstm_net['l_lstm'], l2)
            qembd = lasagne.layers.get_output(q_lstm_net['l_dense'])
        q_type_net = self.build_qn_classifier_mlp(qembd)
        q_type_pred = lasagne.layers.get_output(q_type_net['l_out'],deterministic=False)
        l2_penalty_mlp = regularize_layer_params(q_type_net['l_out'], l2)
        loss = lasagne.objectives.categorical_crossentropy(q_type_pred, qtype)
        loss = loss.mean() + l2_penalty_mlp
        loss += l2_penalty_qlstm
        params = []
        qmlp_params = lasagne.layers.get_all_params(q_type_net['l_out'])
        for p in qmlp_params:
            params.append(p)
        for p in qlstm_params:
            params.append(p)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]

        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.003)
        qtype_test_pred = lasagne.layers.get_output(q_type_net['l_out'],deterministic=True)
        qtype_test_pred = T.argmax(qtype_test_pred, axis=1)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        if from_scratch:
            train = theano.function([qX,mask, qtype], loss, updates=updates, allow_input_downcast=True)
            qtype_predict = theano.function([qX,mask], qtype_test_pred, allow_input_downcast=True)
        else:
            train = theano.function([qembd, qtype], loss, updates=updates, allow_input_downcast=True)
            qtype_predict = theano.function([qembd], qtype_test_pred, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling qtype model..."
        return train, qtype_predict

    def train(self):
        atrain, apredict = self.build_model()
        qtrain, qpredict = self.build_qn_type_model(from_scratch=True)
        num_training_divisions  = int(self.num_division *self.config['train_data_percent']/100)
        num_val_divisions       = num_training_divisions
        self.timer.set_checkpoint('param_save')
        l_loss,l_t_acc,l_v_acc = [],[],[]
        print "Training qn classifier"
        for epoch in range(2):#self.config['epochs']):
            print '\nEpoch :', epoch
            l_loss,l_t_acc,l_v_acc = [],[],[]
            for div_id in range(num_training_divisions):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train')
                qloss = qtrain(qn, mask, qtypes)
                pred_qtypes = qpredict(qn,mask)
                acc = np.mean(pred_qtypes == qtypes)
                l_t_acc.append(acc*100.0)
            for div_id in range(num_val_divisions):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='val')
                pred_qtypes = qpredict(qn,mask)
                l_v_acc.append(100.0*np.mean(pred_qtypes == qtypes))
            l_t_acc, l_v_acc = np.mean(np.array(l_t_acc)), np.mean(np.array(l_v_acc))
            print "train acc %", l_t_acc
            print "val acc %", l_v_acc
        
        print "Training VQA"
        l_loss,l_t_acc,l_v_acc = [],[],[]
        for epoch in range(self.config['epochs']):
            train_accuracy,total = 0,0
            self.timer.set_checkpoint('train') 
            if self.timer.expired('param_save', self.config['checkpoint_interval']):
                self.dump_current_params()
                self.timer.set_checkpoint('param_save')
            loss = []
            for div_id in range(num_training_divisions):
                for a_type in range(len(self.ans_type_dict)):
                    qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train', a_type=a_type)
                    loss.append(self.train_util(qn, mask, iX, ans, sparse_ids, atrain, apredict))
            loss = np.mean(np.array(loss))
            l_loss.append(loss)
            print"Epoch                 : ",epoch
            print"cross_entropy         : %f, time taken (mins) : %f"%(loss,self.timer.print_checkpoint('train'))
        val_acc,train_acc = [],[]
        self.timer.set_checkpoint('val')
        for division_id in range(num_val_divisions):
            qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(division_id, mode='val')
            val_acc.append(self.acc_util(qn, mask, iX, ans, apredict, qpredict))
            qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(division_id, mode='train')
            train_acc.append(self.acc_util(qn, mask, iX, ans, apredict, qpredict))           
        val_acc   = np.mean(np.array(val_acc))
        train_acc = np.mean(np.array(train_acc))
        print"Training accuracy     : %f, time taken (mins) : %f"%(train_acc ,self.timer.print_checkpoint('val') )   
        print"Val accuracy          : %f, time taken (mins) : %f\n"%(val_acc   ,self.timer.print_checkpoint('val') )
        l_v_acc.append(val_acc)
        l_t_acc.append(train_acc)
        self.exp_saver.append_array([l_loss,l_t_acc, l_v_acc],fid='loss_t_v_acc') 

    def train_util(self, qn, mask, iX, Y, sparse_ids, train, predict):
        loss = train(qn, mask, iX, Y, sparse_ids)
        return loss
    
    def acc_util(self, qn, mask, iX, Y, apredict, qpredict):
        pred_qtypes = qpredict(qn, mask)
        pred = []
        for itr,qtype in enumerate(pred_qtypes[:100]):
            temp_pred = apredict(qn[itr:itr+1], mask[itr:itr+1], iX[itr:itr+1], self.ans_per_type[qtype])
            pred.append(self.ans_per_type[qtype][temp_pred])
        pred = np.array(pred)
        acc = np.mean(pred == Y)*100.0
        return acc


    #********* SANITY *********
    
    def sanity_check_train(self):
        atrain, apredict = self.build_model()
        qtrain, qpredict = self.build_qn_type_model(from_scratch=True)
        l_loss,l_acc = [],[]
        div_id = 2
        for i in range(150):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train')
                acc = self.toy_qn_train_util(qn, mask, qtypes, qtrain, qpredict)
                l.print_overwrite("Acc % :", acc * 100.0)
                
        for i in range(200):
            for atype in range(0,len(self.ans_type_dict)):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train', a_type=atype)
                loss, acc = self.toy_train_util(i,qn, mask, iX, ans, qtypes, sparse_ids, atrain, qtrain, qpredict, apredict) 
                
            l_loss.append(loss)
            l_acc.append(acc)
        
    def toy_qn_train_util( self,qX, qtypes, qtrain, qpredict):
        mb = 2000
        qloss = qtrain(qX[:mb], qtypes[:mb])
        pred_qtypes = qpredict(qX[:mb])
        return np.mean(pred_qtypes == qtypes[:mb])

    def toy_train_util(self,epoch, qX, mask, iX, Y, qtype, sparse_i, atrain, qtrain, qpredict, apredict):
        mb = 400
        aloss,qembd = atrain(qX[:mb], mask[:mb], iX[:mb], Y[:mb], sparse_ids)
        pred_qtypes = qpredict(qX[:mb],mask[:mb])
        pred = []
        Y_val = []
        for itr,pq in enumerate(pred_qtypes):
            ans = apredict(qX[itr:itr+1], mask[itr:itr+1], iX[itr:itr+1], self.ans_per_type[pq]) 
            pred.append( self.ans_per_type[pq][ans] )
            Y_val.append( self.ans_per_type[qtype[0]][Y[:mb][itr]] )
        pred = np.array(pred)
        Y_val = np.array(Y_val)
        print "Epoch         : ", epoch
        print "cross_entropy : ", aloss
        print "train acc     : ", 100.0*np.mean(pred==Y_val)
        print "predict      ", [(self.aword[a],a) for a in pred[:10]]
        print "ground truth ", [(self.aword[a],a) for a in Y_val[:10]]
        
        return aloss, 100.0*np.mean(pred==Y[:mb])   
    #************************************************************

    #            TRAINING / VAL DATA RETRIVAL APIS     

    #************************************************************

    def get_data(self, division_num ,mode,a_type=None):
        qn, mask    = self.get_question_data(division_num, mode)    
        ans         = self.get_answer_data(division_num, mode)
        iX          = self.get_image_features(division_num, mode)
        qtypes      = self.get_answer_types(division_num, mode)
        #qn = self.get_one_hot(qn, one_hot_size=len(self.qvocab))
        """
        print "Training data shapes ..."
        print "Question     : ",qn.shape
        print "mask         : ",mask.shape
        print "image feature: ",iX.shape
        print "ans          : ",ans.shape
        """
        sparse_ids = None 
        if a_type is not None:
            yn_ids = [ itr for itr,a in enumerate(ans) if a in self.ans_per_type[a_type]]
            ans = [ self.ans_per_type[a_type].index(a) for a in ans if a in self.ans_per_type[a_type]]
            qn, mask, iX = qn[yn_ids], mask[yn_ids], iX[yn_ids] 
            qtypes, sparse_ids = np.ones(len(yn_ids))*int(a_type), self.ans_per_type[a_type]
        
        if 'yes' in self.config['experiment_id'].lower():
            yn_ids = [ itr for itr,i in enumerate(ans) if self.aword[i] != 'yes' and self.aword[i] != 'no']
            qn, mask, iX, ans = qn[yn_ids], mask[yn_ids], iX[yn_ids], ans[yn_ids]
        
        if 'norm' in self.config['experiment_id'].lower():
            print "normlized input"
            norms = 1.0/np.linalg.norm(iX,axis=1)
            print norms.shape, norms[:10]
            print iX.shape, iX[0,:10]
            iX = np.dot(np.diag(norms), iX)
            print iX.shape, iX[0,:10]
            
        return qn, mask, iX, ans, qtypes, sparse_ids
    
    def get_answer_types(self,division_id,mode):
        s,e = self.divisions[mode][division_id]
        return self.answer_types[mode][s:e]
 
    def get_one_hot(self, qn, one_hot_size):
        output = np.zeros((qn.shape[0], qn.shape[1], one_hot_size), dtype='uint8')
        for i in range(qn.shape[0]):
            output[i] = load_data.one_hot(qn[i],one_hot_size)
        return output
 
    def get_question_data(self, division_id, mode):
        return self.questions[mode][division_id] \
             , self.mask[mode][division_id]

    def get_answer_data(self, division_id, mode):
        s,e = self.divisions[mode][division_id]
        return self.answers[mode][s:e]

    def get_image_features(self, division_id, mode):
        f2l = str(mode) + '_feature' + str(division_id+1) + ".npy"
        f2l = os.path.join(self.config["vqa_model_folder"], f2l)
        return (np.load(f2l)).astype(np.float32)

    def store_question_data(self, mode, load_from_file, save_image_features=False):
        if load_from_file:
            self.questions[mode], self.mask[mode] = self.saver.load_array(fid=str(mode)+'qn_mask')
            return
        qdict = self.qdict[mode]
        question_ids = self.question_ids[mode]
        divisions = self.divisions[mode]
        qn_output = []
        mask_output = []
        for s,e in divisions:
            q_a = np.ones((len(question_ids[s:e]), self.max_qlen), dtype='uint32')*-1
            mask = np.zeros((len(question_ids[s:e]), self.max_qlen), dtype='uint32')
            for itr,q_id in enumerate(question_ids[s:e]):
                q = qdict[q_id]['question']
                l_a = [ self.qvocab[w.lower()] for w in self.tokenizer.tokenize(str(q)) if w.lower() in self.qvocab.keys() ]
                q_a[itr,:len(l_a)] = np.array(l_a[:self.max_qlen], dtype='uint32')
                mask[itr,:len(l_a)] = 1
            qn_output.append(q_a)
            mask_output.append(mask)
        self.questions[mode]    = np.asarray(qn_output)
        self.mask[mode]         = np.asarray(mask_output)
        print "questions shape      :", self.questions[mode].shape
        print "mask shape           :", self.mask[mode].shape
        self.saver.save_array([ self.questions[mode], self.mask[mode] ], fid=str(mode)+'qn_mask')
        if save_image_features:
            self.save_image_data(self.image_ids[mode], self.num_division ,mode)

    def get_image_question_ans_ids(self,mode,load_from_file):
        if load_from_file:
            return self.saver.load_array(fid=str(mode)+"ids")
        image_ids, question_ids = [],[]
        answer_types, answers   = [],[]
        self.ans_type_dict = {} # key : qn_type, value : id
        ans_types = 0
        for a_id,a in enumerate(self.id_info[mode]['top_k_ids']):
            answers.append(a['ans_id'])
            image_ids.append(a['im_id'])
            question_ids.append(a['qn_id'])
            if a['ans_type'] not in self.ans_type_dict.keys():
                self.ans_type_dict[a['ans_type']] = ans_types
                ans_types += 1
            answer_types.append(self.ans_type_dict[a['ans_type']])
        pprint( self.ans_type_dict)
        assert len(image_ids) == len(question_ids)
        assert len(image_ids) == len(answers)

        self.saver.save_array([image_ids, question_ids, answer_types, answers], fid=str(mode)+"ids")
        return np.array(image_ids), np.array(question_ids), np.array(answer_types), np.array(answers)
    
    def answer_type_util(self,mode):
        ans_types = len(self.ans_type_dict)
        self.ans_per_type = dict([(i,[]) for i in range(ans_types)])
        non_numeric_ans = []
        for itr,a_type in enumerate(self.answer_types[mode]):
            a = self.answers[mode][itr]
            isdigit = (self.aword[a]).isdigit()
            yes_no =  (self.aword[a]).lower() == 'yes' or (self.aword[a]).lower() == 'no'
            if yes_no:
                a_type = self.ans_type_dict['yes/no']
                self.answer_types[mode][itr] = a_type
            else:
                if isdigit:
                    a_type = self.ans_type_dict['number']
                    self.answer_types[mode][itr] = a_type
                else:
                    a_type = self.ans_type_dict['other']
                    self.answer_types[mode][itr] = a_type
            if a not in self.ans_per_type[a_type]:
                self.ans_per_type[a_type].append(a)
            """
            if isdigit:
                a_type = self.ans_type_dict['number']
                self.answer_types[mode][itr] = a_type
            if not isdigit and a_type == self.ans_type_dict['number']:
                a_type = self.ans_type_dict['other']
                self.answer_types[mode][itr] = a_type
            if yes_no:
                a_type = self.ans_type_dict['yes/no']
                self.answer_types[mode][itr] = a_type      
            if a not in self.ans_per_type[a_type]:
                self.ans_per_type[a_type].append(a)
            """    
        """
        # hacky coz dataset is messed up
        self.ans_per_type[1].append( self.ans_per_type[0][-1] )
        #self.ans_per_type[1].append( non_numeric_ans )
        self.ans_per_type[0] = self.ans_per_type[0][:2]
        """
        
        print 'yes',[self.aword[a] for a in self.ans_per_type[0]] 
        print 'other',[self.aword[a] for a in self.ans_per_type[1][:20]]
        """
        print 'number',[self.aword[a] for a in self.ans_per_type[2]]
        print 554 in self.ans_per_type[1]
        print 864 in self.ans_per_type[1]
        """
        for a_type in range(ans_types):
            self.ans_per_type[a_type] = sorted(self.ans_per_type[a_type])

    def get_image_file_id(self,image_id,image_db=None):
        for file_id, ilist in enumerate(image_db):
            try:
                loc = ilist.index(image_id)
                return file_id+1, loc
            except ValueError:
                continue
        #print "Image id %d not found" % (image_id)
        return 1,0

    def get_image_data(self,image_ids, mode):
        """
        Output a array of image_features correspodining to the image_ids
        """
        feature_file_template = str(mode) +"_feature"
        image_db = np.load(os.path.join( self.config['cleaned_images_folder'],str(mode) +"_image.npy"))
        print image_db.shape
        im_features = []
        for itr,im_id in enumerate(image_ids):
            file_id, im_loc = self.get_image_file_id(im_id,image_db)
            feature_file = feature_file_template + "_" + str(file_id) + ".npy"
            feature = np.load(os.path.join(self.config['vgg_features_folder'], feature_file))
            im_features.append(feature[im_loc])
            l.print_overwrite("Image data percentage % ", 100*itr/len(image_ids))
        im_features = np.asarray(im_features)
        print "\nfeature shape", im_features.shape
        return im_features
        
    def get_file(self, folder, mode):
        ofiles = os.listdir(folder)
        ofile = os.path.join(folder, [i for i in ofiles if str(i).find(mode) != -1 ][0])  
        return ofile

    def save_image_data(self, image_ids, num_files, mode):
        mbsize = len(image_ids) // num_files
        i = 0
        for s,e in zip(range(0, len(image_ids), mbsize), range(mbsize, len(image_ids), mbsize)):
            i += 1
            print "Saving images from %d to %d" %(s,e)
            f2s = os.path.join(self.config["vqa_model_folder"], str(mode) + '_feature' + str(i))
            np.save(f2s,self.get_image_data(image_ids[s:e],mode))
            print "Saving features to %s ..."%str(f2s)
    
    def unigram_dist(self):
        hist = {}
        total_ans = 0
        for mode in ['train']:
            total_ans += len(self.answers[mode])
            for a in self.answers[mode]:
                if a in hist.keys():
                    hist[a] +=1
                else:
                    hist[a] = 1
        hist = dict([ (k,float(v)/total_ans) for k,v in hist.items() ])
        output = [ -1.0*np.log(hist[a]) for a in range(len(self.avocab))] 
        return output

    def plot_results(self):
        l_loss,l_t_acc, l_v_acc = self.exp_saver.load_array(fid='loss_t_v_acc')
        self.plot_loss(l_loss)
        self.plot_train_val(l_t_acc,l_v_acc)

    def plot_loss(self,loss):
        self.plotter.basic_plot(plot_id='loss_curve',
                                l_Y=[loss],
                                l_Ylabels=['loss'],
                                Ylabel='NLL(in %)',
                                Xlabel='Epochs',
                                title=self.config['experiment_id'] + ': Negative log liklihood curve')
   
    def plot_train_val(self,t_acc,v_acc):
        self.plotter.basic_plot(plot_id='train_val_acc',
                                l_Y=[t_acc,v_acc],
                                l_Ylabels=['train','valid'],
                                Ylabel='Accuracy(in %)',
                                Xlabel='Epochs',
                                title=self.config['experiment_id'] + ': Train / Val Accuracy curve')
