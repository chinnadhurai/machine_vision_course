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

def get_config():
    config = {}
    config["dpath"]  = os.environ['DATAPATH']
    config["opath"]  = os.environ['OUTPUTPATH']
    config["ntrain"] = 95000 # max is 100000
    config["ntest"]  = 10000  # max is 10000
    config["mini_batch_size"] = 32
    config["pickle_file_location"] = config['opath']+'model.zip'
    config["output_images_location"] = config['opath']
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
        print "Arguments needs either be q1 or q2"
        exit(0)
    config = get_config()
    o_conv_net = conv_net(config)
    if sys.argv[1] == "q1":
        o_conv_net.train()
    elif sys.argv[1] == "q2":
        o_conv_net.q2(get_image())
    else:
        print "Arguments can either be q1 or q2"
        exit(0)



