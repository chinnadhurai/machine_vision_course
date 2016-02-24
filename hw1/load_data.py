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



def one_hot(x,n):
	if type(x) == list:
		x = np.array(x)
	x = x.flatten()
	o_h = np.zeros((len(x),n))
	o_h[np.arange(len(x)),x] = 1
	return o_h

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def mirror_image(X):
    i = 0
    Y = np.ones(shape=X.shape)
    while i < Y.shape[-1]:
        Y[:,:,:,i] = X[:,:,:,-1-i]
        i+=1
    return Y

def load_cifar_10_data(config):
    print "loading data from", config['dpath']
    trX = []
    trY = []
    i = 0
    # training data

    file = os.listdir( config['dpath'] )[0]
    data_dict = unpickle( config['dpath'] + file )
    trX = data_dict['data'].reshape(-1,3,32,32)
    trY = np.array(data_dict['labels'])
    trY = one_hot(trY, 10)
    trX = np.concatenate((trX, mirror_image(trX)), axis=0)
    trY = np.concatenate((trY, one_hot(np.array(data_dict['labels']),10)), axis=0)
    #print "--training data :", file, trX.shape, trY.shape

    for file in os.listdir( config['dpath'] )[1:-1]:
        data_dict = unpickle( config['dpath'] + file )
        trdata = data_dict['data'].reshape(-1,3,32,32)
        trX = np.concatenate((trX, mirror_image(trdata)), axis=0)
        trY = np.concatenate((trY, one_hot(np.array(data_dict['labels']),10)), axis=0)
        trX = np.concatenate((trX, trdata), axis=0)
        trY = np.concatenate((trY, one_hot(np.array(data_dict['labels']),10)), axis=0)
        #print "--training data :", file, trX.shape, trY.shape
        i += 1

    #test data
    file = os.listdir( config['dpath'] )[-1]
    data_dict = unpickle( config['dpath'] + file )
    teX = data_dict['data'].reshape(-1,3,32,32)
    teY = np.array(data_dict['labels'])
    #print "--test data :", file, teX.shape, teY.shape

    trX = trX[0:config['ntrain']]
    trY = trY[0:config['ntrain']]
    teX = teX[0:config['ntest']]
    teY = teY[0:config['ntest']]
    print "*** final training data :", trX.shape, trY.shape
    print "*** final test data :", teX.shape, teY.shape
    print "data loaded..."
    return trX,trY,teX,teY