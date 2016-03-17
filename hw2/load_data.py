__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from scipy.misc import imresize
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
    print "loading data from", config['cifar10_path']
    i = 0
    # training data

    file = os.listdir( config['cifar10_path'] )[0]
    data_dict = unpickle( config['cifar10_path'] + file )
    trX = data_dict['data'].reshape(-1,3,32,32)
    trY = np.array(data_dict['labels'])
    if config['data_augment']:
        trX = np.concatenate((trX, mirror_image(trX)), axis=0)
        trY = np.concatenate((trY, np.array(data_dict['labels'])), axis=0)

    for file in os.listdir( config['cifar10_path'] )[1:-1]:
        data_dict = unpickle( config['cifar10_path'] + file )
        trdata = data_dict['data'].reshape(-1,3,32,32)
        trX = np.concatenate((trX, trdata), axis=0)
        trY = np.concatenate((trY, np.array(data_dict['labels'])), axis=0)
        if config['data_augment']:
	    trX = np.concatenate((trX, mirror_image(trdata)), axis=0)
            trY = np.concatenate((trY, np.array(data_dict['labels'])), axis=0)
        #print "--training data :", file, trX.shape, trY.shape
        if 1 == 0:
            convert_to_image(trdata[4],"normal1.jpg")
            mirrod_image =  (mirror_image1(trdata))[4]
            print mirrod_image.shape
            convert_to_image(mirrod_image,"flipped1.jpg")

        i += 1

    slices = np.arange(50000)
    np.random.shuffle(slices)
    train_slices = slices[:config['ntrain_cifar10']]
    test_slices = slices[config['ntrain_cifar10']:]
    teX = trX[test_slices]
    teY = trY[test_slices]
    trX = trX[train_slices]
    trY = trY[train_slices]
    trY = one_hot(trY,10)
    print "*** final training data :", trX.shape, trY.shape
    print "*** final test data :", teX.shape, teY.shape
    print "CIFAR-10 data loaded..."
    return trX,trY,teX,teY

def load_cifar_100_data(config):
    dir        =  config['cifar100_path']
    print "loading data from", dir
    test_file  = os.listdir(dir)[0]
    train_file = os.listdir(dir)[1]	
    test_dict  = unpickle( dir  + test_file )
    train_dict = unpickle( dir  + train_file )
    trX = train_dict['data'].reshape(-1,3,32,32)
    teX = test_dict['data'].reshape(-1,3,32,32)
    if config['fine_labels']:
        nlabels = 100
        label_key = 'fine_labels'
        trY = train_dict['fine_labels']
        trY = one_hot(trY,nlabels)
        teY = test_dict['fine_labels']
        teY = one_hot(teY,nlabels)
    else:
        nlabels = 20
        label_key = 'coarse_labels'
        trY = train_dict['coarse_labels']
        trY = one_hot(trY,nlabels)
        teY = test_dict['coarse_labels']
        teY = one_hot(teY,nlabels)

    trY = train_dict[label_key]
    trY = one_hot(trY,nlabels)
    teY = test_dict[label_key]
    teY = one_hot(teY,nlabels)
    
    slices = np.arange(config['ntrain_cifar100'])
    np.random.shuffle(slices)
    trX = trX[slices]
    trY = trY[slices]
    teX = teX[0:config['ntest_cifar100']]
    teY = teY[0:config['ntest_cifar100']]
    print "*** final training data :", trX.shape, trY.shape
    print "*** final test data :", teX.shape, teY.shape
    print "CIFAR-100 data loaded..."
    return trX,trY,teX,teY	

def upsample(X):
    u_shape = 224
    Y = np.zeros((X.shape[0], X.shape[1], u_shape, u_shape))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i][j] = imresize(X[i][j],(u_shape,u_shape),interp='bilinear', mode=None)
    return Y

def load_cifar_10_data_upsampled(config):
    print "loading data from", config['cifar10_path']
    i = 0
    print "Upsampling..."   
    # training data
    file = os.listdir( config['cifar10_path'] )[0]
    data_dict = unpickle( config['cifar10_path'] + file )
    trX = upsample(data_dict['data'].reshape(-1,3,32,32))
    trY = np.array(data_dict['labels'])
    for file in os.listdir( config['cifar10_path'] )[1:-1]: 
        data_dict = unpickle( config['cifar10_path'] + file )  
        trdata = data_dict['data'].reshape(-1,3,32,32)
        trX = np.concatenate((trX, upsample(trdata)), axis=0)
        trY = np.concatenate((trY, np.array(data_dict['labels'])), axis=0)
    
    #test data
    file = os.listdir( config['cifar10_path'] )[-1]
    data_dict = unpickle( config['cifar10_path'] + file )
    teX = upsample(data_dict['data'].reshape(-1,3,32,32))
    teY = np.array(data_dict['labels'])
    slices = np.arange(config['ntrain_cifar10'])
    np.random.shuffle(slices)
    trX = trX[slices]
    trY = trY[slices]
    teX = teX[:config['ntest_cifar10']]
    teY = teY[:config['ntest_cifar10']]
    print "*** final training data :", trX.shape, trY.shape
    print "*** final test data :", teX.shape, teY.shape
    print "data loaded..."
    return trX,trY,teX,teY

	




