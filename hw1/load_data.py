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

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


def load_data(config):
    print "loading data from", config["dpath"]
    data_dict = {}
    for file in os.listdir(config["dpath"]):
        data_dict[file] = unpickle(config["dpath"] + file)
    print "----",data_dict.keys()
    print "data loaded..."
    return data_dict