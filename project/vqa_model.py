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
import json

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
        self.num_division               = 50
        pprint(config)
        print "\n----------------------"
        print "\n Preping data set..."
        pfile = os.path.join(self.config['questions_folder'], "qvocab.zip")
        self.qvocab, self.qword, self.max_qlen = pickle.load( gzip.open( pfile, "rb" ) )
        pfile = os.path.join(self.config['annotations_folder'], "ans_vocab.zip")
        self.avocab, self.aword = pickle.load( gzip.open( pfile, "rb" ) )
        print "Answer vocab size    :", len(self.avocab)
        print "question vocab size  :", len(self.qvocab)
        self.qdict = {}
        self.image_ids = {}
        self.question_ids = {}
        self.answer_ids = {}
        self.answers = {}
        self.questions = {}
        self.mask= {}
        self.divisions = {}
        for mode in ['train','val']:
            self.qdict[mode] = load_data.load_questions(self.config['questions_folder'], mode=mode)
            self.image_ids[mode], self.question_ids[mode], self.answer_ids[mode] = self.get_image_question_ans_ids(mode)
            mbsize = len(self.image_ids[mode]) // self.num_division
            self.divisions[mode] = zip(range(0, len(self.image_ids[mode]), mbsize), range(mbsize, len(self.image_ids[mode]), mbsize))
            self.store_question_data(mode)
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
        #print len(params)
        #print [ p.get_value().shape for p in params ]
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
        num_training_divisions = int(self.num_division *self.config['train_data_percent']/100)
        for epoch in range(self.config['epochs']):
            l.print_overwrite("epoch :",epoch)
            for division_id in range(num_training_divisions):
                #print " epoch percent done",*100/num_training_divisions)
                qn, mask, iX, ans = self.get_data(division_id, mode='train')
                #self.train_util(qn, mask, iX, ans, train, predict)
        
    def train_util(self, qX, mask, iX, Y, train, predict):
        mb_size = self.batch_size
        for s,e in zip( range(0, len(qX), mb_size), range(mb_size, len(qX), mb_size)):
            loss = train(qX[s:e], mask[s:e], iX[s:e], Y[s:e])
        cumsum = 0
        for s,e in zip( range(0, len(qX), mb_size), range(mb_size, len(qX), mb_size)):
            mask = self.get_mask(qX[s:e])
            pred = predict(qX[s:e], mask ,iX[s:e])
            cumsum += np.sum(pred == Y[s:e])
        print "Training accuracy(in  % )           :", cumsum*100 / Y.shape[0]       

    

    #************************************************************

    #            TRAINING / VAL DATA RETRIVAL APIS     

    #************************************************************

    def get_data(self, division_num ,mode):
        image_ids, question_ids, answer_ids = self.image_ids[mode], self.question_ids[mode], self.answer_ids[mode]
        qn, mask    = self.get_question_data(division_num, mode)    
        ans         = self.get_answer_data(division_num, mode)
        iX          = self.get_image_features(division_num, mode)
        print "Training data shapes ..."
        print "Question     : ",qn.shape
        print "mask         : ",mask.shape
        print "image feature: ",iX.shape
        print "ans          : ",ans.shape
        return qn, mask, iX, ans
    
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

    def store_question_data(self, mode, save_image_features=False):
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
                l_a = [ self.qvocab[w] for w in nltk.word_tokenize(str(q)) ]
                q_a[itr,:len(l_a)] = np.array(l_a, dtype='uint32')
                mask[itr,:len(l_a)] = 1
            qn_output.append(q_a)
            mask_output.append(mask)
        self.questions[mode]    = np.asarray(qn_output)
        self.mask[mode]         = np.asarray(mask_output)
        print "questions shape      :", self.questions[mode].shape
        print "mask shape           :", self.mask[mode].shape
        if save_image_features:
            self.save_image_data(self.image_ids[mode], self.num_division ,mode)

    def get_image_question_ans_ids(self,mode):
        afile = self.get_file(self.config["annotations_folder"],mode=mode)
        answers = json.load(open(afile, 'r'))['annotations']
        self.answers[mode] = []
        image_ids = []
        question_ids = []
        answer_ids = []
        for a_id,a in enumerate(answers):
            qa = nltk.word_tokenize(str(a['multiple_choice_answer']))
            if not len(qa)==1:
                continue
            answer_ids.append(a_id)
            self.answers[mode].append(self.avocab[qa[0]])
            image_ids.append(a['image_id'])
            question_ids.append(a['question_id'])
        self.answers[mode] = np.asarray(self.answers[mode])
        assert len(image_ids) == len(question_ids)
        assert len(image_ids) == len(answer_ids)

        return image_ids, question_ids, answer_ids

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
            np.save(f2s,self.get_image_data(image_ids[s:e]),mode)
            print "Saving features to %s ..."%str(f2s)




