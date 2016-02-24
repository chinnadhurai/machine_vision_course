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
def get_config():
    config = {}
    config["dpath"]  = os.environ['DATAPATH']
    config["opath"]  = os.environ['OUTPUTPATH']
    config["ntrain"] = 1000#95000 # max is 100000
    config["ntest"]  = 100#5000  # max is 10000
    config["mini_batch_size"] = 128
    return config


if __name__ == "__main__":
    config = get_config()
    o_conv_net = conv_net(config)
    o_conv_net.train()



