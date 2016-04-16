__author__ = 'chinna'

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from theano.misc.pkl_utils import dump
from theano.misc.pkl_utils import load
from scipy.misc import imread
from PIL import Image
import os
import cPickle as pickle
import gzip
import h5py
import sys
sys.dont_write_bytecode = True

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        #acc = theano.shared(p.get_value() * 0.)
        #acc_new = rho * acc + (1 - rho) * g ** 2
        #gradient_scaling = T.sqrt(acc_new + epsilon)
        #g = g / gradient_scaling
        #updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates,grads

def print_overwrite(string,val):
    import sys
    sys.stdout.write('\r' + string + str(val))
    sys.stdout.flush()

def dump_params_pickle(file_path,params_to_pickle):
    print "Dumping params to ",file_path 
    with open(file_path, 'wb') as f:
        pickle.dump(params_to_pickle, f)

def load_params_pickle(file_path):
    print "loading params from",file_path
    with open(file_path, 'rb') as f:
        loaded_params = pickle.load(f)
    return loaded_params

def load_params_pickle_gzip(file_path):
    print "loading params from",file_path
    with open(file_path, 'rb') as f:
        loaded_params = pickle.load(f)
    return loaded_params

def get_kernel(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def convert_to_image(image, name):
    import matplotlib
    matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
    import matplotlib.pyplot as plt
    #plt.imshow(image, interpolation='none')
    #plt.savefig(name)
    print "saving file : ",name
    a = (255*image).astype('uint8')
    im = Image.fromarray(a)
    im.save(name)

def add_gnoise_util(image):
    from scipy import signal
    kernel = get_kernel(shape=(10,10),sigma=5)
    image   = signal.convolve2d(image, kernel, boundary='fill', fillvalue=0,mode='same')
    #print(image.shape)
    return image

def add_gnoise_util_kernel(image,kernel):
    from scipy import signal
    image   = signal.convolve2d(image, kernel, boundary='fill', fillvalue=0,mode='same')
    return image

def get_dir(home_dir,arg):
    final_dir=os.path.join(home_dir,arg)
    if not os.path.exists(final_dir):
        os.makedirs(final_dir)
    return final_dir

def get_file(dir_name, filename):
    return os.path.join(dir_name, filename)

def dump_h5(filepath, data):
    print "h5py : dumping data into", filepath 
    h5f = h5py.File(filepath, 'w')
    for i in range(len(data)):
        h5f.create_dataset('dataset_'+str(i), data=data[i])
    h5f.close()

def load_h5(filepath):
    print "h5py : loading data from", filepath
    h5f = h5py.File(filepath,'r')
    params = []
    for key in h5f.keys():
        params.append(h5f[key][:])
    h5f.close()
    return params

def make_folder(directory):
    return
