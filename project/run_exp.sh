#!/bin/bash
source init_setup.sh
THEANO_FLAGS="floatX=float32,device=gpu6" python run_exp.py vqa_train
