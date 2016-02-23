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


def load_cifar_10_data(config):
    print "loading data from", config['dpath']
    trX = {}
    trY = {}
    i = 0
    # training data
    for file in os.listdir( config['dpath'] )[:-1]:
        data_dict = unpickle( config['dpath'] + file )
        trX[i]    = data_dict['data'].reshape(-1,3,32,32)
        trY[i]    = np.array(data_dict['labels'])
        print "--training data :", file, trX[i].shape, trY[i].shape
        i += 1

    #test data
    file = os.listdir( config['dpath'] )[-1]
    data_dict = unpickle( config['dpath'] + file )
    teX = data_dict['data'].reshape(-1,3,32,32)
    teY = np.array(data_dict['labels'])
    print "--test data :", file, teX.shape, teY.shape
    print "data loaded..."
    return trX,trY,teX,teY