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
    config['fine_tune_vgg']             = False
    config['train_data_percent']        = 10
    config['epochs']                    = 1
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Arguments needs either be chunk / dummy / q1 or q2, NUM_TRAINING, ALPHA"
        exit(0)
    elif sys.argv[1] == "vocab":
        config = get_config() 
        ifolder = os.path.join( config["dpath"],"real_images/")
        l.get_vocab(ifolder)
    elif sys.argv[1] == "chunk":
        config = get_config() 
        ifolder = os.path.join( config["dpath"],"real_images/")    
        afolder = os.path.join( ifolder,"annotations")
        qfolder = os.path.join( ifolder,"questions")
        modes = ['train','val','test']
        for mode in modes:
            v,w = l.load_coco_data(ifolder, os.path.join(ifolder, "cleaned_images"), mode=mode)
    elif sys.argv[1] == "dummy":
        config = get_config() 
        ifolder = os.path.join( config["dpath"],"real_images")            
        afolder = os.path.join( ifolder,"annotations")                      
        qfolder = os.path.join( ifolder,"questions") 
        #l.load_annotations(afolder)
        l.get_answer_vocab(afolder)
        #l.vqa_api(qfolder,afolder)
        #l.get_question_vocab(qfolder)
    elif sys.argv[1] == "gen_vgg_features":
        config = get_config()
        vgg_feature_extractor = vgg_feature(config)
        features_folder = config['vgg_features_folder']
        image_array_folder = config['image_array_folder']
        vgg_feature_extractor.create_vgg_feature_dataset(image_array_folder, features_folder)      
    elif sys.argv[1] == 'build_lstm':
        config = get_config()
        vqa_classifier = vqa_type(config)
        vqa_classifier.train()
        #vqa_classifier.get_train_data()
    else:
        print "Arguments can either be q1 or q2"
        exit(0)

