from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
import numpy
import random
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score
import theano

from keras import backend as K

import numpy as np
import scipy as sp
from scipy import signal
import os
from PIL import Image

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

from scipy.misc import imread


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


def add_gnoise_util(image):
    kernel = get_kernel(shape=(10,10),sigma=3)
    image   = signal.convolve2d(image, kernel, boundary='fill', fillvalue=0,mode='same')
    #print(image.shape)
    return image


def convert_to_image(image, name):

	#plt.imshow(image, interpolation='none')
	#plt.savefig(name)
    a = (image*255).astype('uint8')
    im = Image.fromarray(a)
    im.save(name)


im = imread("boat.jpg")

im = numpy.swapaxes(im,1,2)
im = numpy.swapaxes(im,0,1)

im = im[numpy.newaxis,:,:,:]
print(im.shape)


im = im.astype('float32')
#im /= 255


batch_size = 256
nb_classes = 10
nb_epoch = 200
data_augmentation = True


# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()


model = Sequential()

model.add(Convolution2D(64, 3, 3, border_mode='same',
                        input_shape=(img_channels, img_rows, img_cols)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(128, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(Convolution2D(256, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Convolution2D(1024, 4, 4, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution2D(1024, 1, 1, border_mode='valid'))
model.add(Activation('relu'))

model.add(Convolution2D(nb_classes, 1, 1, border_mode='valid'))
vc = BatchNormalization()
model.add(vc)
model.add(Flatten())
#model.add(Dense(nb_classes))
model.add(Activation('softmax'))


model.load_weights("/data/lisatmp4/sarath/data/output/conv/1/weights.hdf5")#/data/lisatmp4/chinna/data/ift6268/temp/1/weights.hdf5")

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)



convout = theano.function([model.get_input(train=False)], vc.get_output(train=False))
t0 = time.clock()
[layer_output] = convout(im)
print(layer_output.shape)


dpath = "/data/lisatmp4/chinna/data/ift6268/temp/1/"

for i in range(0,10):
	convert_to_image(layer_output[i],dpath+str(i)+"old.jpg")
	layer_output[i] = add_gnoise_util(layer_output[i])
	print(max(layer_output[i].flatten()))
	convert_to_image(layer_output[i],dpath+str(i)+".jpg")	
print ("Time")
print (time.clock() - t0)
