#!/bin/bash
source init_setup.sh
THEANO_FLAGS="floatX=float32,device=gpu6,lib.cnmem=0.85" python run_exp.py q1
THEANO_FLAGS="floatX=float32,device=gpu6,lib.cnmem=0.85" python run_exp.py q2
