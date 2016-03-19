#!/bin/bash
source init_setup.sh
for alpha in 0.3 0.5 0.7 0.9
do
  for n in 10000 20000 15000
  do
    THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q1a $n $alpha 
    THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q1b $n $alpha
  done
done
rm -rf /data/lisatmp4/chinna/data/ift6268/logging/plotq*
