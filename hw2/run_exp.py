__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from PIL import Image
import os
from q1 import conv_net
from scipy.misc import imread
import sys
import load_data as l
import lib 
import vgg_16
from q2 import conv_classifier_type
import gzip 
import h5py
sys.dont_write_bytecode = True

# arg 1 ->  q1 
# arg 2 ->  n
# arg 3 -> alpha
def get_config_q1(is_transfer_learning):
    config = {}
    config["dpath"]                     = os.environ['DATAPATH']
    config["opath"]                     = os.environ['OUTPUTPATH']
    config["cifar10_path"]              = config["dpath"] + "cifar10/"
    config["cifar100_path"]             = config["dpath"] + "cifar100/cifar-100-python/"
    config["plt_path"]                  = lib.get_dir(config["opath"],"plot"+str(sys.argv[1])+"_"+str(sys.argv[2])+"_"+str(sys.argv[3]))
    config["ntrain_cifar10"]            = min(49000,int(sys.argv[2])) 	# max is 50000
    config["ntest_cifar10"]             = 50000 - config["ntrain_cifar10"]  	# max is 10000
    config["data_augment"]              = False
    config["transfer_learning"]         = is_transfer_learning
    config["ntrain_cifar100"]           = 50000 	# max is 50000
    config["ntest_cifar100"]            = 10000 	# max is 10000
    config["fine_labels"]               = False   #True     
    config["mini_batch_size"]           = 32
    config["pickle_file_location"]      = config['opath']+'model.zip'
    config["output_images_location"]    = config['opath'] + 'figs/'
    config['epochs']                    = 25
    config["alpha"]                     = max(0.1,float(sys.argv[3]))
    config["plt_path"]                  = lib.get_dir(config["opath"],"plots"+str(sys.argv[1])+"_"+str(config["alpha"]))
    config["cifar10_plt_file"]          = lib.get_file(config["plt_path"], "plot_"+str(is_transfer_learning) +"_" +str(config["ntrain_cifar10"]) + "_cifar10.jpg")
    config["cifar100_plt_file"]          = lib.get_file(config["plt_path"], "plot_"+str(is_transfer_learning) +"_" +str(config["ntrain_cifar10"]) + "_cifar100.jpg")
    return config

def get_config_q2():
    config = {}
    config["dpath"]                     = os.environ['DATAPATH']
    config["opath"]                     = os.environ['OUTPUTPATH']
    config["cifar10_path"]              = config["dpath"] + "cifar10/"
    config["data_augment"]              = False
    config["ntrain_cifar10"]            = 50000
    config['ntest_cifar10']             = 10000
    config['params']                    = os.environ['DATAPATH'] + "vgg_params/vgg16.pkl"
    config['mini_batch_size']           = 256
    config['epochs']                    = 50
    config['load_upsampled_frm_pkl']    = False
    config['upsample_pkl_file']         = os.environ['DATAPATH'] + "upsampled.h5"   
    config['dataset_file']              = os.environ['DATAPATH'] + "dataset.h5"#"dataset_custom_kernel.h5"
    config['load_dataset_file']         = True #False
    config["lamda_list"]                = [0.0,1e-6,1e-5,1e-4,2e-4,1e-3]
    return config

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print "Arguments needs either be dummy or q1 or q2, NUM_TRAINING, ALPHA"
        exit(0)
    if sys.argv[1] == "q1a":
        config = get_config_q1(True)
        o_conv_net = conv_net(config)
	o_conv_net.train()
    elif sys.argv[1] == "q1b":
        config = get_config_q1(False)
	o_conv_net = conv_net(config)
        o_conv_net.train()
    elif sys.argv[1] == "q2":
        config = get_config_q2()
        classifier = conv_classifier_type(config)
        classifier.train()
    elif sys.argv[1] == "gen_picture":
        config = get_config_q1(True)
        l.load_cifar_100_data(config,config["dpath"] + "cifar100.jpeg") 
    elif sys.argv[1] == "dummy":
        x,y = np.zeros((1,3)), np.zeros(50)
        dfile = os.environ['DATAPATH'] + "dataset1.h5"
        lib.dump_h5(dfile,[x,y])
        a,b = lib.load_h5(dfile)
        print a.shape, b.shape
    else:
        print "Arguments can either be q1 or q2"
        exit(0)

