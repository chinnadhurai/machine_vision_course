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
    config["dpath"] = os.environ['DATAPATH']
    config["opath"] = os.environ['OUTPUTPATH']
    return config


if __name__ == "__main__":
    config = get_config()
    o_conv_net = conv_net(config)



