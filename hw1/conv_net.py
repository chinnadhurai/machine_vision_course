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
import cPickle as pickle
from load_data import load_data

class conv_net:
    def __init__(self, config):
        self.config = config
        self.data = load_data(config)
        print "created conv_net"
