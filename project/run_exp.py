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


def get_config():
    config = {}
    config["dpath"]                     = os.environ['DATAPATH']
    config["opath"]                     = os.environ['OUTPUTPATH']
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Arguments needs either be dummy or q1 or q2, NUM_TRAINING, ALPHA"
        exit(0)
    elif sys.argv[1] == "dummy":
        config = get_config()     
        afolder = os.path.join( config["dpath"],"real_images/annotations")
        qfolder = os.path.join( config["dpath"],"real_images/questions")
        start = 0
        mode = 'TRAIN'
        #l.load_coco_data(imfolder, imfolder, mode=mode)
        l.load_annotations(afolder)
    else:
        print "Arguments can either be q1 or q2"
        exit(0)

