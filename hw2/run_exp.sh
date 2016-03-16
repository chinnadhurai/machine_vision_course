#!/bin/bash
source init_setup.sh
for alpha in 0.3 0.5 0.7 0.9
do
  THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q1 1000 $alpha 
  THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q2 1000 $alpha
  THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q1 2000 $alpha
  THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q2 2000 $alpha
  THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q1 5000 $alpha
  THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q2 5000 $alpha
done
rm -rf /data/lisatmp4/chinna/data/ift6268/logging/plotq*
