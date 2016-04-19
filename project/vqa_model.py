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
        self.params                     = []
        pprint(config)
        print "\n----------------------"
        print "\nPreping data set..."
        self.timer = l.timer_type()
        self.saver = l.save_np_arrays( os.path.join(self.config['questions_folder'], "temp"))
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
        self.qdict = {}
        self.image_ids = {}
        self.question_ids = {}
        self.answer_type = {}
        self.answers = {}
        self.questions = {}
        self.mask= {}
        self.divisions = {}
        self.saved_params = {}
        self.timer.set_checkpoint('init')
        load_from_file= True
        for mode in ['train','val']:
            self.qdict[mode] = load_data.load_questions(self.config['questions_folder'], mode=mode)
            self.image_ids[mode], self.question_ids[mode], self.answer_type[mode], self.answers[mode] = self.get_image_question_ans_ids(mode, load_from_file=load_from_file)
            mbsize = len(self.image_ids[mode]) // self.num_division
            self.divisions[mode] = zip(range(0, len(self.image_ids[mode]), mbsize), range(mbsize, len(self.image_ids[mode]), mbsize))
            self.store_question_data(mode, load_from_file=load_from_file, save_image_features=False)
        print "init time taken", self.timer.print_checkpoint('init')
        self.p_ans_vector = self.unigram_dist()
        print "Initialization done ..."
        
    def add_to_param_list(self,network,l_params,param_type):
        for p in l_params:
            self.params.append(p)
        local_dict = {}
        local_dict['params'] = l_params
        uid = self.timer.get_uid()
        local_dict['f2s'] = os.path.join(self.config['saved_params'], str(param_type) + "_params")
        self.saved_params[param_type] = local_dict
        self.load_saved_params(network,param_type)
    
    def dump_current_params(self):
        print "Saving current params to ", self.config['saved_params']
        params = []
        for k,v in self.saved_params.items():
            params = [p.get_value() for p in v['params']]
            self.saver.save_array(params,fid=str(k) + '_model_params')

    def load_saved_params(self, network, param_type):
        if not self.config['load_from_saved_params']:
            return
        print "Loading %s params from %s"%(param_type,self.saved_params[param_type]['f2s'])
        params = self.saver.load_array(fid=str(k) + '_model_params')
        lasagne.layers.set_all_param_values(network, params)

    def load_vgg_params(self):
        param_loc = self.config['vgg_params']
        params = l.load_params_pickle(param_loc)
        return params['param values']
    
    def build_question_boW(self, input_var, mask=None):
        return
        

    def build_question_lstm(self, input_var, mask=None):
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
        return image_feature * question_feature
    
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
        
        net['l_h2_drop'] = lasagne.layers.DropoutLayer(net['l_h2'], p=0.5)
        
        net['l_out'] = lasagne.layers.DenseLayer(
                net['l_h2_drop'],
                num_units=len(self.avocab),
                nonlinearity=lasagne.nonlinearities.softmax)
                #W = lasagne.init.Constant(0.),
                #b = lasagne.init.Categorical(self.p_ans_vector))
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
        qX, mask, Y = self.qX, self.lstm_mask, self.Y
        q_lstm_net = self.build_question_lstm(qX, mask)
        ql_out = lasagne.layers.get_output(q_lstm_net['l_dense'])
        vgg_mlp_net = self.build_vgg_feature_mlp(iX)
        vgg_out = lasagne.layers.get_output(vgg_mlp_net['l_out'])
        mlp_input = self.combine_image_question_model(ql_out, vgg_out)
        network = self.build_mlp_model(mlp_input)['l_out']
        prediction = lasagne.layers.get_output(network, deterministic=False)
        loss = lasagne.objectives.categorical_crossentropy(prediction, Y)
        loss = loss.mean()
        params = self.params#lasagne.layers.get_all_params(network)
        all_grads = T.grad(loss, params)
        if self.grad_clip != None:
            all_grads = [T.clip(g, self.grad_clip[0], self.grad_clip[1]) for g in all_grads]
        #print len(params)
        #print [ p.get_value().shape for p in params ]
        updates = lasagne.updates.adam(all_grads, params, learning_rate=0.01)
        #updates = lasagne.updates.nesterov_momentum(
        #    loss, params, learning_rate=0.01, momentum=0.9)
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
        test_prediction = T.argmax(test_prediction, axis=1)
        print "Compiling..."
        self.timer.set_checkpoint('compile')
        train = theano.function([qX, mask, iX, Y], loss, updates=updates, allow_input_downcast=True)
        predict = theano.function([qX, mask, iX], test_prediction, allow_input_downcast=True)
        print "Compile time(mins)", self.timer.print_checkpoint('compile')
        print "Done Compiling final model..."
        return train,predict
    
    def train(self):
        train, predict = self.build_model()
        num_training_divisions  = int(self.num_division *self.config['train_data_percent']/100)
        num_val_divisions       = num_training_divisions
        self.timer.set_checkpoint('param_save')
        for epoch in range(self.config['epochs']):
            train_accuracy,total = 0,0
            self.timer.set_checkpoint('train') 
            if self.timer.expired('param_save', self.config['checkpoint_interval']):
                self.dump_current_params()
                self.timer.set_checkpoint('param_save')
            loss = []
            for division_id in range(num_training_divisions):
                qn, mask, iX, ans = self.get_data(division_id, mode='train')
                loss.append(self.train_util(qn, mask, iX, ans, train, predict))
            loss = np.mean(np.array(loss))
            print"Epoch                : ",epoch
            print"cross_entropy        : %f, time taken (mins) : %f"%(loss,self.timer.print_checkpoint('train'))
            val_acc,train_acc = [],[]
            self.timer.set_checkpoint('val')
            for division_id in range(num_val_divisions):
                qn, mask, iX, ans = self.get_data(division_id, mode='val')
                val_acc.append(self.acc_util(qn, mask, iX, ans, train, predict))
                qn, mask, iX, ans = self.get_data(division_id, mode='train')
                train_acc.append(self.acc_util(qn, mask, iX, ans, train, predict))           
            val_acc = np.mean(np.array(val_acc))*100.0
            train_acc = np.mean(np.array(train_acc))*100.0
            print"Training accuracy     : %f, time taken (mins) : %f"%(train_acc ,self.timer.print_checkpoint('val') )   
            print"Val accuracy          : %f, time taken (mins) : %f\n"%(val_acc   ,self.timer.print_checkpoint('val') )
    
    def train_util(self, qX, mask, iX, Y, train, predict):
        mb_size = self.config['batch_size']
        loss = []
        for s,e in zip( range(0, len(qX), mb_size), range(mb_size, len(qX), mb_size)):
            loss.append(train(qX[s:e], mask[s:e], iX[s:e], Y[s:e]))
        return np.mean(np.array(loss))
    
    def acc_util(self, qX, mask, iX, Y, train, predict):
        mb_size = self.config['batch_size'] 
        cumsum,total = 0.0,0.0
        for s,e in zip( range(0, len(qX), mb_size), range(mb_size, len(qX), mb_size)):
            pred = predict(qX[s:e], mask[s:e] ,iX[s:e])
            cumsum += np.sum(pred == Y[s:e])
            total += len(pred)
            #print [(self.aword[a],a) for a in pred[:2]]
            #print [(self.aword[a],a) for a in Y[s:s+2]]
        return cumsum/total

    def sanity_check_train(self):
        train, predict = self.build_model()
        qn, mask, iX, ans = self.get_data(2, mode='train', sanity_mode=True)
        for i in range(1000):
            self.toy_train_util(i,qn, mask, iX, ans,train, predict)    
    
    def toy_train_util(self,epoch, qX, mask, iX, Y, train, predict):
        mb = 2000
        loss = train(qX[:mb], mask[:mb], iX[:mb], Y[:mb])
        pred = predict(qX[:mb], mask[:mb], iX[:mb])
        print "Epoch         : ", epoch
        print "cross_entropy : ", loss
        print "train acc     : ", 100.0*np.mean(pred==Y[:mb])
        print "predict      ", [(self.aword[a],a) for a in pred[:10]]
        print "ground truth ", [(self.aword[a],a) for a in Y[:10]]
            
    #************************************************************

    #            TRAINING / VAL DATA RETRIVAL APIS     

    #************************************************************

    def get_data(self, division_num ,mode, sanity_mode='False'):
        qn, mask    = self.get_question_data(division_num, mode)    
        ans         = self.get_answer_data(division_num, mode)
        iX          = self.get_image_features(division_num, mode)
        #qn = self.get_one_hot(qn, one_hot_size=len(self.qvocab))
        """
        print "Training data shapes ..."
        print "Question     : ",qn.shape
        print "mask         : ",mask.shape
        print "image feature: ",iX.shape
        print "ans          : ",ans.shape
        """
        yn_ids = [ itr for itr,i in enumerate(ans) if self.aword[i] != 'yes' and self.aword[i] != 'no']
        qn, mask, iX, ans = qn[yn_ids], mask[yn_ids], iX[yn_ids], ans[yn_ids]
        
        return qn, mask, iX, ans
    
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
        return np.load(f2l)

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
        for a_id,a in enumerate(self.id_info[mode]['top_k_ids']):
            answers.append(a['ans_id'])
            answer_types.append(a['ans_type'])
            image_ids.append(a['im_id'])
            question_ids.append(a['qn_id'])
        
        assert len(image_ids) == len(question_ids)
        assert len(image_ids) == len(answers)

        self.saver.save_array([image_ids, question_ids, answer_types, answers], fid=str(mode)+"ids")
        return np.array(image_ids), np.array(question_ids), np.array(answer_types), np.array(answers)
    
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
