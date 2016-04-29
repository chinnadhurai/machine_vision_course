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
from model import MODEL

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
        self.ql_out                     = T.fmatrix()
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
        self.model = MODEL( config = config,
                            qvocab_len=len(self.qvocab),
                            num_ans=len(self.avocab),
                            num_qtypes=len(self.ans_type_dict),
                            l_saver=[self.saver, self.exp_saver] )

        print "Initialization done ..."
        

    def train_qn_classifier(self):
        print "Training qn classifier"
        qtrain, qpredict, qembd_fn = self.model.build_qn_type_model()
        num_training_divisions  = int(self.num_division *self.config['train_data_percent']/100)
        num_val_divisions       = num_training_divisions
        for epoch in range(3):
            print '\nEpoch :', epoch 
            l_loss,l_t_acc,l_v_acc = [],[],[]
            for div_id in np.shuffle(np.arange(num_training_divisions)):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train')
                qloss = qtrain(qn, mask, qtypes)
                acc = qpredict(qn,mask)
                #acc = np.mean(pred_qtypes == qtypes)
                l_t_acc.append(acc*100.0)
            for div_id in range(num_val_divisions):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='val')
                acc = qpredict(qn,mask)
                l_v_acc.append(100.0*acc)
            l_t_acc, l_v_acc = np.mean(np.array(l_t_acc)), np.mean(np.array(l_v_acc))
            print "train acc %", l_t_acc
            print "val acc %", l_v_acc
        print "Done training Question classifier\n"
        return qembd_fn

    def train(self):
        #qembd_fn = self.train_qn_classifier()
        atrain, apredict = self.model.build_vqa_model_vanilla()
        self.timer.set_checkpoint('param_save')
        
        print "Training VQA"
        l_loss,l_t_acc,l_v_acc = [],[],[]
        for epoch in range(self.config['epochs']):
            train_accuracy,total = 0,0
            self.timer.set_checkpoint('train') 
            if self.timer.expired('param_save', self.config['checkpoint_interval']):
                self.dump_current_params()
                self.timer.set_checkpoint('param_save')
            loss = self.train_util(atrain)
            loss = np.mean(np.array(loss))
            l_loss.append(loss)
            print"Epoch                 : ",epoch
            print"cross_entropy         : %f, time taken (mins) : %f"%(loss,self.timer.print_checkpoint('train'))
            self.timer.set_checkpoint('val')
            val_acc = self.acc_util('val',apredict)
            train_acc =  self.acc_util('train',apredict)
            print"Training accuracy     : %f, time taken (mins) : %f"%(train_acc ,self.timer.print_checkpoint('val') )   
            print"Val accuracy          : %f, time taken (mins) : %f\n"%(val_acc   ,self.timer.print_checkpoint('val') )

    def train_util(self, train):
        loss,l_div_ids,l_atypes = [],range(self.num_division),range(len(self.ans_type_dict))
        np.random.shuffle(l_div_ids)
        np.random.shuffle(l_atypes)
        for div_id in l_div_ids:
            for a_type in l_atypes:
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train', a_type=a_type)
            #qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train')
                loss.append(train(qn, mask, iX, ans))
        return np.mean(np.array(loss))
    
    def acc_util(self,mode,apredict):
        acc, l_div_ids = [], range(self.num_division)
        np.random.shuffle(l_div_ids)
        for division_id in l_div_ids:
            qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(division_id, mode=mode)
            acc.append(apredict(qn, mask, iX, ans)*100.0)
        """
        for itr,qtype in enumerate(pred_qtypes):
            temp_pred = apredict(qembd[itr:itr+1], iX[itr:itr+1], self.ans_per_type[qtype])
            #temp_pred = apredict(qn[itr:itr+1], mask[itr:itr+1], iX[itr:itr+1], self.ans_per_type[qtype])
            #pred.append(self.ans_per_type[qtype][temp_pred])
            pred.append(
        pred = np.array(pred)
        """
        return np.mean(np.array(acc))

    #************************************************************

    #            TRAINING / VAL DATA RETRIVAL APIS     

    #************************************************************
    def get_data(self, division_num ,mode,a_type=None):
        qn, mask    = self.get_question_data(division_num, mode)    
        ans         = self.get_answer_data(division_num, mode)
        iX          = self.get_image_features(division_num, mode)
        qtypes      = self.get_answer_types(division_num, mode)
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
            qtypes, sparse_ids = np.ones(len(yn_ids))*int(a_type), np.array(self.ans_per_type[a_type])
        
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
        
        for a_type in range(ans_types):
            self.ans_per_type[a_type] = sorted(self.ans_per_type[a_type])

    
    def get_answer_types(self,division_id,mode):
        s,e = self.divisions[mode][division_id]
        return self.answer_types[mode][s:e]
 
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
            load_data.save_image_data(self.config, self.image_ids[mode], self.num_division ,mode)

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
