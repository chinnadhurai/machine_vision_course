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
import skipthoughts


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
        self.q_type_dict                = {'Multiple-Choice' : 0, 'Open-Ended':1}
        pprint(config)
        print "\n----------------------"
        print "\nPreping data set..."
        self.timer = l.timer_type()
        self.saver = l.save_np_arrays(os.path.join(self.config['questions_folder'], "temp"))
        self.exp_saver = l.save_np_arrays(os.path.join(self.config['real_abstract_images'] + "/models", config['experiment_id']))
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
        self.answers, self.questions, self.mask, self.qn_cat, self.divisions, self.saved_params = {},{},{},{},{},{}
        self.timer.set_checkpoint('init')
        load_from_file= True
        for mode in ['train','val']:
            self.qdict[mode] = load_data.load_questions(self.config['questions_folder'], mode=mode, cat=config['qn_category'])
            self.image_ids[mode], self.question_ids[mode], self.answer_types[mode], self.answers[mode] = self.get_image_question_ans_ids(mode, load_from_file=load_from_file)
            mbsize = len(self.image_ids[mode]) // self.num_division
            self.divisions[mode] = zip(range(0, len(self.image_ids[mode]), mbsize), range(mbsize, len(self.image_ids[mode]), mbsize))
            self.store_st_qn_data(mode, load_from_file=load_from_file, save_image_features=False)
            self.answer_type_util(mode=mode)
        print "Init time taken(in mins)", self.timer.print_checkpoint('init')
        self.model = MODEL( config = config,
                            qvocab_len=len(self.qvocab),
                            max_qlen=self.max_qlen,
                            num_ans=len(self.avocab),
                            num_qtypes=len(self.ans_type_dict),
                            l_saver=[self.saver, self.exp_saver] )

        print "Initialization done ..."
        

    def train_qn_classifier(self,epochs):
        print "Training qn classifier"
        qtrain, qpredict, qembd_fn = self.model.build_qn_type_model()
        num_training_divisions  = int(self.num_division *self.config['train_data_percent']/100)
        num_val_divisions       = num_training_divisions
        for epoch in range(epochs):
            print '\nEpoch :', epoch 
            l_loss,l_t_acc,l_v_acc = [],[],[]
            for div_id in np.arange(num_training_divisions):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train')
                qloss = qtrain(qn, mask, qtypes)
                qtypes, acc = qpredict(qn,mask,qtypes)
                l_t_acc.append(acc*100.0)
            for div_id in range(num_val_divisions):
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='val')
                qtypes, acc = qpredict(qn,mask,qtypes)
                l_v_acc.append(100.0*acc)
            l_t_acc, l_v_acc = np.mean(np.array(l_t_acc)), np.mean(np.array(l_v_acc))
            print "train acc %", l_t_acc
            print "val acc %", l_v_acc
        print "Done training Question classifier\n"
        return qpredict, qembd_fn

    def train(self):
        qtype_predict, qembd_fn = self.train_qn_classifier(epochs=0)
        atrain, apredict = self.model.build_vqa_model_skip_thought_conv()
        self.timer.set_checkpoint('param_save')
        epoch, no_improv, patience, best_val_acc = 0,0,10,0
        print "Training VQA"
        l_loss,l_t_acc,l_v_acc = [],[],[]
        while no_improv <= patience:
            epoch += 1
            self.timer.set_checkpoint('train') 
            if self.timer.expired('param_save', self.config['checkpoint_interval']):
                self.dump_current_params()
                self.timer.set_checkpoint('param_save')
            loss = self.train_util_vanilla(atrain,qembd_fn)
            loss = np.mean(np.array(loss))
            l_loss.append(loss)
            print"Epoch                 : ",epoch
            print"cross_entropy         : %f, time taken (mins) : %f"%(loss,self.timer.print_checkpoint('train'))
            self.timer.set_checkpoint('val')
            val_acc = self.acc_util('val',apredict, qtype_predict,qembd_fn)
            train_acc =  self.acc_util('train',apredict, qtype_predict,qembd_fn)
            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                no_improv = 0
            else:
                no_improv += 1
            print"Training accuracy     : %f, time taken (mins) : %f"%(train_acc ,self.timer.print_checkpoint('val') )   
            print"Val accuracy          : %f, time taken (mins) : %f\n"%(val_acc   ,self.timer.print_checkpoint('val') )
            
    def train_util(self, train, qembd_fn):
        loss,l_div_ids_atypes = [],np.arange(self.num_division*len(self.ans_type_dict))
        np.random.shuffle(l_div_ids_atypes)
        for i in range(2000):
            for itr in l_div_ids_atypes:
                div_id,a_type = divmod(itr, len(self.ans_type_dict))
                qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train', a_type=a_type)
                #qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train')
                try:
                    loss.append(train(qn[i:i+1], mask[i:i+1], iX[i:i+1], ans[i:i+1], sparse_ids))
                except IndexError:
                    print "skipping ",i
                    continue
        return np.mean(np.array(loss))

    def train_util_vanilla(self, train, qembd_fn):
        loss,l_div_ids = [],np.arange(self.num_division)
        np.random.shuffle(l_div_ids)
        for div_id in l_div_ids:
            qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(div_id, mode='train')
            #qembd = qembd_fn(qn, mask)
            loss.append(train(qn, iX, ans))
        return np.mean(np.array(loss))
    
    def acc_util(self,mode,apredict, qtype_predict, qembd_fn):
        acc, l_div_ids = [], np.arange(self.num_division)
        np.random.shuffle(l_div_ids)
        for division_id in l_div_ids:
            qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(division_id, mode=mode)
            #qembd = qembd_fn(qn, mask)
            acc.append(apredict(qn, iX, ans)*100.0)
        return np.mean(np.array(acc))

    def acc_util_sparse_ids(self,mode,apredict, qtype_predict, qembd_fn):
        acc = []
        division_id = np.random.randint(49)
        for a_type in range(len(self.ans_type_dict)):
            qn, mask, iX, ans, qtypes, sparse_ids = self.get_data(division_id, mode=mode,a_type=a_type)
            pred_qtypes,mode_acc = qtype_predict(qn,mask,qtypes)
            for itr,qtype in enumerate(pred_qtypes):
                temp_acc = apredict(qn[itr:itr+1], mask[itr:itr+1], iX[itr:itr+1], ans[itr:itr+1], self.ans_per_type[qtype])
                acc.append(temp_acc)
        return np.mean(np.array(acc))*100.0

    #************************************************************

    #            TRAINING / VAL DATA RETRIVAL APIS     

    #************************************************************
    def get_data(self, division_num ,mode,a_type=None):
        qn                  = self.get_question_data(division_num, mode)    
        ans                 = self.get_answer_data(division_num, mode)
        iX                  = self.get_image_features(division_num, mode)
        qtypes              = self.get_answer_types(division_num, mode)
        mask, sparse_ids = None, None
        
        if a_type is not None:
            yn_ids = [ itr for itr,a in enumerate(ans) if a in self.ans_per_type[a_type]]
            ans = [ self.ans_per_type[a_type].index(a) for a in ans if a in self.ans_per_type[a_type]]
            #ans = ans[yn_ids]
            qn, mask, iX = qn[yn_ids], mask[yn_ids], iX[yn_ids] 
            qtypes, sparse_ids = np.ones(len(yn_ids))*int(a_type), np.array(self.ans_per_type[a_type], dtype=np.float32)
        
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
        return self.get_st_qn_data(mode,division_id)
        #return self.questions[mode][division_id] 
             #, self.mask[mode][division_id]

    def get_answer_data(self, division_id, mode):
        s,e = self.divisions[mode][division_id]
        return self.answers[mode][s:e]

    def get_image_features(self, division_id, mode):
        f2l = str(mode) + '_feature' + str(division_id+1) + ".npy"
        f2l = os.path.join(self.config["vqa_model_folder"], f2l)
        return (np.load(f2l)).astype(np.float32)

    def store_question_data(self, mode, load_from_file, save_image_features=False):
        if load_from_file:
            self.questions[mode], self.mask[mode],  self.qn_cat[mode] = self.saver.load_array(fid=str(mode)+'qn_mask')
            return
        qdict = self.qdict[mode]
        question_ids = self.question_ids[mode]
        divisions = self.divisions[mode]
        qn_output = []
        mask_output = []
        qn_cat_output = []
        for s,e in divisions:
            q_a = np.ones((len(question_ids[s:e]), self.max_qlen), dtype='uint32')*-1
            mask = np.zeros((len(question_ids[s:e]), self.max_qlen), dtype='uint32')
            q_cat = np.ones((len(question_ids[s:e])), dtype='uint32')*-1
            for itr,q_id in enumerate(question_ids[s:e]):
                q = qdict[q_id]['question']
                q_cat[itr] = self.q_type_dict[qdict[q_id]['type']]
                l_a = [ self.qvocab[w.lower()] for w in self.tokenizer.tokenize(str(q)) if w.lower() in self.qvocab.keys() ]
                q_a[itr,:len(l_a)] = np.array(l_a[:self.max_qlen], dtype='uint32')
                mask[itr,:len(l_a)] = 1
            qn_output.append(q_a)
            mask_output.append(mask)
            qn_cat_output.append(q_cat)
        self.questions[mode]    = np.asarray(qn_output)
        self.mask[mode]         = np.asarray(mask_output)
        self.qn_cat[mode]       = np.asarray(qn_cat_output)
        print "questions shape      :", self.questions[mode].shape
        print "mask shape           :", self.mask[mode].shape
        print "qcat shape           :", self.qn_cat[mode].shape
        self.saver.save_array([ self.questions[mode], self.mask[mode], self.qn_cat[mode]], fid=str(mode)+'qn_mask')
        if save_image_features:
            load_data.save_image_data(self.config, self.image_ids[mode], self.num_division ,mode)

    def store_st_qn_data(self, mode, load_from_file, save_image_features=False):
        if load_from_file:
            return
        model = skipthoughts.load_model()
        l_qembd,divisions,question_ids = [], self.divisions[mode], self.question_ids[mode]
        div_id = 0
        for s,e in divisions[:2]:
            l_q = [ self.qdict[mode][q_id]['question'] for q_id in question_ids[s:e] ]
            qembd = skipthoughts.encode(model, l_q, verbose=False)[:,:2400]
            self.saver.save_array([qembd], fid=str(mode)+str(div_id)+'skip_thought_qembd')
            div_id += 1    
        if save_image_features:
            load_data.save_image_data(self.config, self.image_ids[mode], self.num_division ,mode)
    
    def get_st_qn_data(self, mode, div_id):
        return self.saver.load_array(fid=str(mode)+str(div_id)+'skip_thought_qembd')[0]

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
