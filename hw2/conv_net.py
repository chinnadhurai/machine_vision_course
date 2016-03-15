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
from load_data import load_cifar_10_data,load_cifar_100_data
import lib as l
#from theano.tensor.nnet import conv2d
from theano.sandbox.cuda.dnn import dnn_conv as conv2d
from theano.tensor.signal.pool import pool_2d as max_pool_2d
import sys
from scipy.misc import imread

class conv_net:
    def __init__(self, config):
        self.config = config
        print "Experiment Configuration:"
        print "Num cifar-10 training examples       : ", config["ntrain_cifar10"]
        print "Num cifar-10 test examples           : ", config["ntest_cifar10"]
        print "Num cifar-100 training examples      : ", config["ntrain_cifar100"]
        print "Num cifar-100 test examples          : ", config["ntest_cifar100"]
        print "Minibatch size                       : ", config["mini_batch_size"]
        print "Using fine labels                    : ", config["fine_labels"]
        print "Alpha                                : ", config["alpha"]
        self.trX10, self.trY10, self.teX10, self.teY10 = load_cifar_10_data(config)
        self.trX100, self.trY100, self.teX100, self.teY100 = load_cifar_100_data(config)
        self.X = T.ftensor4()
        self.Y = T.fmatrix()
        self.alpha = config["alpha"]
       
        if config["fine_labels"]:
            self.cifar10_labels = 100
        else:
            self.cifar10_labels = 20

        #weights init ( output depth/filers x input depth x filter_h x filter_w
        self.w1  = l.init_weights((64, 3, 3, 3))      #conv
        self.w2  = l.init_weights((128, 64, 3, 3))    #conv
        self.w3  = l.init_weights((256, 128, 3, 3))   #conv
        self.w4  = l.init_weights((256, 256, 3, 3))   #conv
        self.w5  = l.init_weights((1024,256,1,1))     #full-conn
        self.w6  = l.init_weights((1024,1024,1,1))    #full-conn
        self.w_o10 = l.init_weights((10,1024,1,1))      #full-conn
        self.w_o100 = l.init_weights((self.cifar10_labels,1024,1,1))      #full-conn
          
        self.b1  = theano.shared(np.asarray(np.zeros((1,64,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))
        self.b2  = theano.shared(np.asarray(np.zeros((1,128,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))
        self.b3  = theano.shared(np.asarray(np.zeros((1,256,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))
        self.b4  = theano.shared(np.asarray(np.zeros((1,256,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))
        self.b5  = theano.shared(np.asarray(np.zeros((1,1024,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))
        self.b6  = theano.shared(np.asarray(np.zeros((1,1024,1,1)), dtype=theano.config.floatX),broadcastable=(True,False,True,True))

        #batch_norm params
        self.b10         = theano.shared(np.zeros((1,10)),broadcastable=(True,False))
        self.g10         = theano.shared(np.ones((1,10)),broadcastable=(True,False))
        self.r_m10       = theano.shared(np.zeros((1,10)),broadcastable=(True,False))
        self.r_s10       = theano.shared(np.zeros((1,10)),broadcastable=(True,False))
        
        self.b100        = theano.shared(np.zeros((1,self.cifar10_labels)),broadcastable=(True,False))
        self.g100        = theano.shared(np.ones((1,self.cifar10_labels)),broadcastable=(True,False))
        self.r_m100      = theano.shared(np.zeros((1,self.cifar10_labels)),broadcastable=(True,False))
        self.r_s100      = theano.shared(np.zeros((1,self.cifar10_labels)),broadcastable=(True,False))
        self.params_to_pickle = [self.w1, self.w2, self.w3, self.w4, self.w5, self.w6, self.w_o10, self.w_o100, self.b1,self.b2,self.b3,self.b4,self.b5,self.b6,self.r_m10, self.r_s10,self.g10, self.b10, self.r_m100, self.r_s100,self.g100,self.b100]

        print "Initializing and building conv_net"

    def model(self, X, w1, w2, w3, w4, w5, w6, w_o, g, b):
        l1a = l.rectify(conv2d(X, w1, border_mode='valid') + self.b1)
        l1 = max_pool_2d(l1a, (2, 2), ignore_border=True)
        
        l2a = l.rectify(conv2d(l1, w2,border_mode='valid') + self.b2)
        l2 = max_pool_2d(l2a, (2, 2), ignore_border=True)

        l3 = l.rectify(conv2d(l2, w3, border_mode='valid') + self.b3)

        l4a = l.rectify(conv2d(l3, w4, border_mode='valid') + self.b4)
        l4 = max_pool_2d(l4a, (2, 2), ignore_border=True)
        
        l5 = l.rectify(conv2d(l4, w5, border_mode='valid') + self.b5)
       
        l6 = l.rectify(conv2d(l5, w6, border_mode='valid') + self.b6)
        l6 = conv2d(l6, w_o, border_mode='valid')
        l6 = T.flatten(l6, outdim=2)
        l6 = ((l6 - T.mean(l6, axis=0))/(T.std(l6,axis=0)+1e-4))*g + b
        pyx = T.nnet.softmax(l6)
        return l1, l2, l3, l4, l5, l6, pyx

    def update_running_mean_std(self, updates, r_m, r_s, i_m, i_s, a = 0.99):
        updates.append((r_m, a*r_m + (1-a)*i_m ))
        updates.append((r_s, a*r_s + (1-a)*i_s ))

    def test_model(self, X, w1, w2, w3, w4, w5, w6, w_o, r_m, r_s, g, b):
        l1a = l.rectify(conv2d(X, w1, border_mode='valid') + self.b1)
        l1 = max_pool_2d(l1a, (2, 2), ignore_border=True)

        l2a = l.rectify(conv2d(l1, w2,border_mode='valid') + self.b2)
        l2 = max_pool_2d(l2a, (2, 2), ignore_border=True)

        l3 = l.rectify(conv2d(l2, w3, border_mode='valid') + self.b3)

        l4a = l.rectify(conv2d(l3, w4, border_mode='valid') + self.b4)
        l4 = max_pool_2d(l4a, (2, 2), ignore_border=True)

        l5 = l.rectify(conv2d(l4, w5, border_mode='valid') + self.b5)

        l6 = l.rectify(conv2d(l5, w6, border_mode='valid') + self.b6)
        l6 = conv2d(l6, w_o, border_mode='valid')
        l6 = T.flatten(l6, outdim=2)
        l6 = ((l6 - r_m)/(r_s + 1e-4))*g + b
        pyx = T.nnet.softmax(l6)
        return pyx

    def build_model(self):
        self.train_cifar10,self.predict_cifar10 = self.build_model_util(self.alpha, self.w_o10, self.g10, self.b10, self.r_m10, self.r_s10)
        print "Done building the cifar10 model..."    
        self.train_cifar100,self.predict_cifar100 = self.build_model_util(1-self.alpha, self.w_o100, self.g100, self.b100, self.r_m100, self.r_s100)
        print "Done building the cifar100 model..."    
    
    def build_model_util(self, alpha, w_o, g, b, r_m, r_s):
        X, Y, w1, w2, w3, w4, w5, w6 = self.X, self.Y, self.w1, self.w2, self.w3, self.w4, self.w5, self.w6
        b1,b2,b3,b4,b5,b6 = self.b1,self.b2,self.b3,self.b4,self.b5,self.b6
        l1, l2, l3, l4, l5, l6, py_x = self.model(X, w1, w2, w3, w4, w5, w6, w_o, g, b)
        cost = alpha * T.mean(T.nnet.categorical_crossentropy(py_x, Y))
        params = [w1, w2, w3, w4, w5, w6, w_o, g, b, b1, b2, b3, b4, b5, b6]
        updates,grads = l.RMSprop(cost, params, lr=0.01)
        self.update_running_mean_std(updates,r_m,r_s,T.mean(l6, axis=0), T.std(l6,axis=0))
        train = theano.function(inputs=[X, Y], outputs=[cost,T.sum((grads)[0]),l1], updates=updates, allow_input_downcast=True)
        py_x = self.test_model(X, w1, w2, w3, w4, w5, w6, w_o, r_m, r_s, g, b)
        y_x = T.argmax(py_x, axis=1)
        predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)
        return train, predict       
        
    def train(self):
        self.build_model()

        trX10, trY10, teX10, teY10 = self.trX10, self.trY10, self.teX10, self.teY10
        trX100, trY100, teX100, teY100 = self.trX100, self.trY100, self.teX100, self.teY100
        mbsize = self.config['mini_batch_size']
        for i in range(self.config['epochs']):
            print "epoch :",i
            for start, end in zip(range(0, len(trX10), mbsize), range(mbsize, len(trX10), mbsize)):
                cost,grads,entropy = self.train_cifar10(trX10[start:end], trY10[start:end])
                l.print_overwrite("cost : ",cost)
            for start, end in zip(range(0, len(trX100), mbsize), range(mbsize, len(trX100), mbsize)):
                cost,grads,entropy = self.train_cifar100(trX100[start:end], trY100[start:end])
                l.print_overwrite("cost : ",cost)
            print "\nCIFAR10 : train accracy :", np.mean(np.argmax(trY10[:5000], axis=1) == self.predict_cifar10(trX10[:5000])) \
            ,"\tvalidation accuracy : ",np.mean(teY10[:10000] == self.predict_cifar10(teX10[:10000]))
	    print "CIFAR100: train accracy :", np.mean(np.argmax(trY100[:5000], axis=1) == self.predict_cifar100(trX100[:5000])) \
            ,"\tvalidation accuracy : ",np.mean(np.argmax(teY100[:10000],axis=1) == self.predict_cifar100(teX100[:10000]))
            l.dump_params_pickle(self.config["pickle_file_location"],self.params_to_pickle)

    #============================================================
    # QUESTION -2
    #============================================================
    def test_model_for_bigger_image(self,X, w1,w2,w3,w4,w5,w6,w_o,b1,b2,b3,b4,b5,b6,r_m,r_s,g,b ):
        l1a = l.rectify(conv2d(X, w1, border_mode='valid') + b1)
        l1 = max_pool_2d(l1a, (2, 2), ignore_border=True)


        l2a = l.rectify(conv2d(l1, w2,border_mode='valid') + b2)
        l2 = max_pool_2d(l2a, (2, 2), ignore_border=True)

        l3 = l.rectify(conv2d(l2, w3, border_mode='valid') + b3)

        l4a = l.rectify(conv2d(l3, w4, border_mode='valid') + b4)
        l4 = max_pool_2d(l4a, (2, 2), ignore_border=True)

        l5 = l.rectify(conv2d(l4, w5, border_mode='valid') + b5)

        l6 = l.rectify(conv2d(l5, w6, border_mode='valid') + b6)
        l6a = conv2d(l6, w_o, border_mode='valid')
        l6 = T.max(l6a,axis=(2,3),keepdims=False)
        l6 = ((l6 - r_m)/(r_s + 1e-4))*g + b

        pyx = T.nnet.softmax(l6)
        return pyx, l6, l6a

    def load_model(self):
        w1,w2,w3,w4,w5,w6,w_o,b1,b2,b3,b4,b5,b6,r_m,r_s,g,b = l.load_params_pickle(self.config["pickle_file_location"])
        py_x,l6,l6a = self.test_model_for_bigger_image(self.X, w1,w2,w3,w4,w5,w6,w_o,b1,b2,b3,b4,b5,b6,r_m,r_s,g,b )
        y_x = T.argmax(py_x, axis=1)
        self.predict = theano.function(inputs=[self.X], outputs=[y_x,l6,l6a], allow_input_downcast=True)

    def q2(self,im):
        self.load_model()
        dpath = self.config["output_images_location"]
        y_x,l6,l6a = self.predict(im)
	l6a = np.asarray(l6a)
	print type(l6a)
        for i in np.arange(l6.shape[1]):
	    p_im = l6a[0,i] / (sum(l6a[0,i].flatten())) 
            p_im = (p_im - np.average(p_im))/np.std(p_im)
            print "max value :", i, max(p_im.flatten())
            smooth_image = l.add_gnoise_util(p_im)
            l.convert_to_image(smooth_image,dpath+"fig_"+str(i)+".jpg")

