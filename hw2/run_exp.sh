#!/bin/bash
source init_setup.sh
for alpha in 0.1  0.9
do
  for n in 1000 5000 10000
  do
    THEANO_FLAGS='floatX=float32,device=gpu6' python run_exp.py q1a $n $alpha 
  done
done

rm -rf /data/lisatmp4/chinna/data/ift6268/logging/plotq*
