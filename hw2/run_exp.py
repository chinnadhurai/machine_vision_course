__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
import matplotlib.pyplot as plt
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

def get_config():
    config = {}
    config["dpath"]                     = os.environ['DATAPATH']
    config["opath"]                     = os.environ['OUTPUTPATH']
    config["cifar10_path"]              = config["dpath"] + "cifar10/"
    config["cifar100_path"]             = config["dpath"] + "cifar100/cifar-100-python/"
    config["ntrain_cifar10"]            = 5000 	# max is 50000
    config["ntest_cifar10"]             = 50000 - config["ntrain_cifar10"]  	# max is 10000
    config["data_augment"]              = False
    config["transfer_learning"]         = True
    config["ntrain_cifar100"]           = 50000 	# max is 50000
    config["ntest_cifar100"]            = 10000 	# max is 10000
    config["fine_labels"]               = False   #True     
    config["mini_batch_size"]           = 32
    config["pickle_file_location"]      = config['opath']+'model.zip'
    config["output_images_location"]    = config['opath'] + 'figs/'
    config['epochs']                    = 45
    config["alpha"]                     = 0.5
    return config

def get_image():
    im = imread("boat.jpg")
    im = np.swapaxes(im,1,2)
    im = np.swapaxes(im,0,1)
    im = im[np.newaxis,:,:,:]
    print(im.shape)
    im = im.astype('float32')
    return im

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Arguments needs either be dummy or q1 or q2"
        exit(0)
    config = get_config()
    if sys.argv[1] == "q1":
        o_conv_net = conv_net(config)
	o_conv_net.train()
    elif sys.argv[1] == "q2":
	o_conv_net = conv_net(config)
        o_conv_net.q2(get_image())
    elif sys.argv[1] == "dummy":
	l.load_cifar_100_data(config)
        l.load_cifar_10_data(config)
    else:
        print "Arguments can either be q1 or q2"
        exit(0)



