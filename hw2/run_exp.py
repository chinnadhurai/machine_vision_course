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
from conv_net import conv_net
from scipy.misc import imread
import sys
import load_data as l
import lib 
import vgg_16
from q2 import conv_classifier_type
 
def get_config(is_transfer_learning):
    config = {}
    config["dpath"]                     = os.environ['DATAPATH']
    config["opath"]                     = os.environ['OUTPUTPATH']
    config["cifar10_path"]              = config["dpath"] + "cifar10/"
    config["cifar100_path"]             = config["dpath"] + "cifar100/cifar-100-python/"
    config["plt_path"]                  = lib.get_dir(config["opath"],"plot"+str(sys.argv[1])+"_"+str(sys.argv[2])+"_"+str(sys.argv[3]))
    config["ntrain_cifar10"]            = min(49000,int(sys.argv[2])) 	# max is 50000
    config["ntest_cifar10"]             = 50000 - config["ntrain_cifar10"]  	# max is 10000
    config["data_augment"]              = False
    config["transfer_learning"]         = is_transfer_learning
    config["ntrain_cifar100"]           = 50000 	# max is 50000
    config["ntest_cifar100"]            = 10000 	# max is 10000
    config["fine_labels"]               = False   #True     
    config["mini_batch_size"]           = 32
    config["pickle_file_location"]      = config['opath']+'model.zip'
    config["output_images_location"]    = config['opath'] + 'figs/'
    config['epochs']                    = 20
    config["alpha"]                     = max(0.1,float(sys.argv[3]))
    config["plt_path"]                  = lib.get_dir(config["opath"],"plots"+str(sys.argv[1])+"_"+str(config["alpha"]))
    config["plt_file"]                  = lib.get_file(config["plt_path"], "plot_"+str(is_transfer_learning) +"_" +str(config["ntrain_cifar10"])+".jpg")
    return config

def get_config_q2():
    config = {}
    config["dpath"]                     = os.environ['DATAPATH']
    config["opath"]                     = os.environ['OUTPUTPATH']
    config["cifar10_path"]              = config["dpath"] + "cifar10/"
    config["data_augment"]              = False
    config["ntrain_cifar10"]            = 49000
    config['ntest_cifar10']             = 10000
    config['params']                    = os.environ['DATAPATH'] + "vgg_params/vgg16.pkl"
    config['mini_batch_size']           = 32
    config['epochs']                    = 10
    
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Arguments needs either be dummy or q1 or q2, NUM_TRAINING, ALPHA"
        exit(0)
    if sys.argv[1] == "q1a":
        config = get_config(True)
        o_conv_net = conv_net(config)
	o_conv_net.train()
    elif sys.argv[1] == "q1b":
        config = get_config(False)
	o_conv_net = conv_net(config)
        o_conv_net.train()
    elif sys.argv[1] == "q2":
        config = get_config_q2()
        classifier = conv_classifier_type(config)
        classifier.train()
    elif sys.argv[1] == "dummy":
	l.load_cifar_100_data(config)
        l.load_cifar_10_data(config)
    else:
        print "Arguments can either be q1 or q2"
        exit(0)

