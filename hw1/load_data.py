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

def mirror_image1(X_train):
    X2_train = X_train.copy()
    count = 0
    for i in range(X2_train.shape[0]):
        for j in range(X2_train[i].shape[0]):
            X2_train[i][j] = np.fliplr(X2_train[i][j])
            count+=1
    print count
    return X2_train

def convert_to_image(image, name):
    image = np.swapaxes(image,0,1)
    image = np.swapaxes(image,1,2)
    im = Image.fromarray(image)
    im.save(name)

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
        trX = np.concatenate((trX, trdata), axis=0)
        trY = np.concatenate((trY, one_hot(np.array(data_dict['labels']),10)), axis=0)
        trX = np.concatenate((trX, mirror_image(trdata)), axis=0)
        trY = np.concatenate((trY, one_hot(np.array(data_dict['labels']),10)), axis=0)
        #print "--training data :", file, trX.shape, trY.shape
        if 1 == 0:
            convert_to_image(trdata[4],"normal1.jpg")
            mirrod_image =  (mirror_image1(trdata))[4]
            print mirrod_image.shape
            convert_to_image(mirrod_image,"flipped1.jpg")

        i += 1

    #test data
    file = os.listdir( config['dpath'] )[-1]
    data_dict = unpickle( config['dpath'] + file )
    teX = data_dict['data'].reshape(-1,3,32,32)
    teY = np.array(data_dict['labels'])
    #print "--test data :", file, teX.shape, teY.shape
    slices = np.arange(config['ntrain'])
    np.random.shuffle(slices)
    print slices[:10]
    trX = trX[slices]
    trY = trY[slices]
    teX = teX[0:config['ntest']]
    teY = teY[0:config['ntest']]
    print "*** final training data :", trX.shape, trY.shape
    print "*** final test data :", teX.shape, teY.shape
    print "data loaded..."
    return trX,trY,teX,teY