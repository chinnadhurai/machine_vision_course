__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image
import os
from scipy.misc import imread
import sys
import load_data as l
import lib 
import vgg_16
import gzip 
import h5py
sys.dont_write_bytecode = True
import pickle
from vgg_model_custom import vgg_feature
from vqa_model import vqa_type

def get_config(image_mode='real'):
    config = {}
    d_image_mode = { 'real':'real_images', 'abstract':'abstract_images'}
    config["dpath"]                     = os.environ['DATAPATH']
    config["opath"]                     = os.environ['OUTPUTPATH']
    config['real_abstract_images']      = os.path.join( config['dpath'], d_image_mode[image_mode])
    config['vgg_params']                = os.path.join( config['dpath'], 'vgg_params/vgg16.pkl')
    config['questions_folder']          = os.path.join( config['real_abstract_images'],'questions')
    config['annotations_folder']        = os.path.join( config['real_abstract_images'],'annotations')    
    config['vgg_features_folder']       = os.path.join( config["real_abstract_images"], 'vgg_features')
    config['vqa_model_folder']          = os.path.join( config["vgg_features_folder"], 'vqa_modelA')
    config['cleaned_images_folder']     = os.path.join( config["real_abstract_images"], 'cleaned_images')
    config['saved_params']              = os.path.join( config['opath'], 'params')
    config['load_from_saved_params']    = False
    config['checkpoint_interval']       = 60 #mins
    config['fine_tune_vgg']             = False
    config['train_data_percent']        = 100
    config['epochs']                    = 100
    config['mlp_input_dim']             = 1024
    config['lstm_hidden_dim']           = 300
    config['bptt_trunk_steps']          = -1
    config['grad_clip']                 = 100
    config['batch_size']                = 128
    config['num_division']              = 50
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Arguments needs either be chunk / vocab / gen_vgg_features / vqa_train" 
        exit(0)
    elif sys.argv[1] == "chunk":
        config = get_config() 
        ifolder = config['real_abstract_images']
        afolder = os.path.join( ifolder,"annotations")
        qfolder = os.path.join( ifolder,"questions")
        modes = ['train','val','test']
        for mode in modes:
            print mode
            l.load_coco_data(ifolder, os.path.join(ifolder, "cleaned_images1"), mode=mode)
    elif sys.argv[1] == "vocab":
        config = get_config() 
        ifolder = os.path.join( config["dpath"],"real_images")            
        afolder = os.path.join( ifolder,"annotations")                      
        qfolder = os.path.join( ifolder,"questions") 
        l.get_answer_vocab(afolder)
        l.get_question_vocab(qfolder)
    elif sys.argv[1] == "gen_vgg_features":
        config = get_config()
        vgg_feature_extractor = vgg_feature(config)
        features_folder = config['vgg_features_folder']
        image_array_folder = config['image_array_folder']
        vgg_feature_extractor.create_vgg_feature_dataset(image_array_folder, features_folder)      
    elif sys.argv[1] == 'vqa_train':
        config = get_config()
        vqa_classifier = vqa_type(config)
        vqa_classifier.train()
    else:
        print "Arguments needs either be chunk / vocab / gen_vgg_features / vqa_train"
        exit(0)

