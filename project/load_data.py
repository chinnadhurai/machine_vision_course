__author__ = 'chinna'
import theano
from theano import tensor as T
import numpy as np
from theano import shared
from theano import function
import scipy as sp
from scipy import signal
from scipy.misc import imresize,imread
from PIL import Image
import os
import lib as l
import cPickle
import sys 
from os import listdir
import os
import h5py
import json
from pprint import pprint
import re
import nltk
from vqa import VQA

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
    print "Saving image to ", name
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
    teY = one_hot(trY[test_slices],10)
    trX = trX[train_slices]
    trY = trY[train_slices]
    trY = one_hot(trY,10)
    print "*** final training data :", trX.shape, trY.shape
    print "*** final test data :", teX.shape, teY.shape
    print "CIFAR-10 data loaded..."
    return trX,trY,teX,teY

def load_cifar_100_data(config, gen_picture_file=None):
    dir        =  config['cifar100_path']
    print "loading data from", dir
    test_file  = os.listdir(dir)[0]
    train_file = os.listdir(dir)[1]	
    test_dict  = unpickle( dir  + test_file )
    train_dict = unpickle( dir  + train_file )
    trX = train_dict['data'].reshape(-1,3,32,32)
    teX = test_dict['data'].reshape(-1,3,32,32)
    
    if gen_picture_file is not None:
        for i in range(30):
            pic_file = gen_picture_file + "cifar100_" + str(i) + ".jpg"
            convert_to_image(trX[np.random.randint(5000)], pic_file)
        return
    
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

def upsample_custom(X):
    print "upsampling custom..."
    u_shape = 224
    i_shape = X.shape[2]
    r = u_shape / i_shape
    Y = np.zeros((X.shape[0], X.shape[1], u_shape, u_shape))
    Y[:,:,::r,::r] = X
    kernel = l.get_kernel(shape=(20,20),sigma=8)
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i][j] = l.add_gnoise_util_kernel(Y[i][j],kernel)
    return Y

def load_cifar_10_data_upsampled(config):
    print "loading data from", config['cifar10_path']
    if config['load_upsampled_frm_pkl']:
        trX,trY,teX,teY = l.load_h5(config['upsample_pkl_file'])
        print "*** Upsampled training data :", trX.shape, trY.shape
        print "*** Upsampled test data :", teX.shape, teY.shape
        print "data loaded..."
        return trX,trY,teX,teY

    i = 0
    print "Upsampling..."   
    # training data
    file = os.listdir( config['cifar10_path'] )[0]
    data_dict = unpickle( config['cifar10_path'] + file )
    upsample = upsample_custom
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
    params_to_pickle = [trX,trY,teX,teY]
    #l.dump_h5(config["upsample_pkl_file"],params_to_pickle)
    return trX,trY,teX,teY


def load_coco_data ( in_folder, \
                     o_folder, \
                     mode = 'train', \
                     num_files = 25,
                     vgg_shape = 224 ):
    mode = mode.lower()
    im_folder = [ d for d in listdir(os.path.join(in_folder,'images')) if d.find(mode.lower()) != -1 ][0]
    im_folder = os.path.join(in_folder,'images/'+str(im_folder))
    imgs = listdir(im_folder)
    mbsize = len(imgs) // num_files
    i = 0
    l_image_id = []
    for start,end in zip(range(0, len(imgs), mbsize), range(mbsize, len(imgs), mbsize)):
        i = i + 1
        filepath = os.path.join(o_folder, mode  + "_image_" + str(i))
        images = []
        image_ids = []
        for im_file in imgs[start:end]:
            if not im_file.endswith('.jpg'):
                continue
            image_ids.append( int(re.findall(r'\d+',im_file)[-1]) )
            image = imread(os.path.join(im_folder,im_file))
            if len(image.shape) == 3:
                image = np.swapaxes(image,1,2)
                image = np.swapaxes(image,0,1)
                t_image = np.zeros((image.shape[0], vgg_shape, vgg_shape))
                for j in range(image.shape[0]):
                    t_image[j] = imresize(image[j],(vgg_shape,vgg_shape),interp='bilinear', mode=None)
                images.append(t_image)
        images = np.asarray(images, dtype=np.uint8)
        np.save(filepath, images)
        l_image_id.append(image_ids)
        print "Writing to ", filepath
    l_image_id = np.asarray(l_image_id)
    filepath = os.path.join(o_folder, mode  + "_image")
    np.save(filepath, l_image_id)
    print "Saving image ids to ", filepath


def get_vocab(folder):
    wc = 0
    vocab = {}
    word = {}
    for s_type in ['annotations', 'questions']:
        s_folder = os.path.join(folder,[fr for fr in listdir(folder) if s_type in str(fr)][0])
        print s_folder
        for f in listdir(s_folder):
            dataset = json.load(open(os.path.join(s_folder,f), 'r'))
            for q in dataset[s_type]:
                for key, item in q.items():
                    print s_type, key 
                    qa =" ".join(re.findall("[a-zA-Z]+", str(item)))
                    for w in qa.lower().split():
                        if w not in vocab:
                            vocab[w] = wc
                            word[wc] = w
                            wc+= 1           
            print "...", f, wc  
    print len(vocab)
    return vocab, word


def get_answer_vocab(folder):
    wc = 0
    vocab = {}
    word = {}
    max_qlen = 0
    oneword_ans = 0
    total_ans = 0
    for f in listdir(folder):
        dataset = json.load(open(os.path.join(folder,f), 'r'))
        for q in dataset['annotations']:
            qa = nltk.word_tokenize(str(q['multiple_choice_answer']))
            oneword_ans += int( len(qa) == 1)
            total_ans += 1
            if max_qlen < len(qa):
                max_qlen = len(qa)
            for w in qa:
                if w not in vocab:
                    vocab[w] = wc
                    word[wc] = w
                    wc+= 1
        print "...", f, wc , max_qlen, oneword_ans, total_ans
    print len(vocab)
    return vocab, word, max_qlen


def get_question_vocab(folder):
    wc = 0
    vocab = {}
    word = {}
    max_qlen = 0
    for f in listdir(folder):
        dataset = json.load(open(os.path.join(folder,f), 'r'))
        for q in dataset['questions']:
            qa = nltk.word_tokenize(str(q['question']))#" ".join(re.findall("[a-zA-Z0-9]+", str(q['question'])))
            if max_qlen < len(qa):
                max_qlen = len(qa)
            for w in qa:
                if w not in vocab:
                    vocab[w] = wc
                    word[wc] = w
                    wc+= 1
        print "...", f, wc , max_qlen
    print len(vocab)
    return vocab, word, max_qlen

def load_annotations(folder, mode='val'):
    afiles = listdir(folder)
    a_file = [i for i in afiles if str(i).find(mode) != -1][0]
    print "Getting annotations from ", a_file
    a_dict = {}     
    dataset = json.load(open(os.path.join(folder,a_file), 'r'))['annotations']
    for d in dataset:
        a_dict[d['question_id']]= d
    print "Number of questions: ", len(a_dict)
    return a_dict

def load_questions(folder, mode='val'):
    qfiles = listdir(folder)
    qfile = os.path.join(folder, [i for i in qfiles if str(i).find(mode) != -1 ][0])
    print "Getting questions from ", qfile
    qdict = {}
    data = json.load(open(qfile, 'r'))
    for q in data['questions']:
        localdict = {}
        # task_type = Multiple-Choice, Open-Ended  
        localdict['type']  = data['task_type']
        localdict['question'] = q['question']
        qdict[q['question_id']] = localdict
    return qdict
    """
    for k,v in data.items():
        print k,"\n... "
        if type(v) == dict:
            print "dict", len(v), v.keys()
        if type(v) == list:
            print "list", len(v), v[0]
        else:
            print v
    """

def vqa_api(qfolder,afolder,mode='train'):
    print "test"
    qfiles = listdir(qfolder)
    afiles = listdir(afolder)
    qfile = os.path.join(qfolder, [i for i in qfiles if str(i).find(mode) != -1 ][0])
    afile = os.path.join(afolder, [i for i in afiles if str(i).find(mode) != -1 ][0])
    vqa = VQA(afile,qfile)
    
    
