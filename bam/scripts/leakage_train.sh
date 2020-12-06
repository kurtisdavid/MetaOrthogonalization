#!/bin/bash

device=3
device1=0
device2=1 #7
device3=4
device4=0
main_epochs=30 #30
n2v_epochs=20 #20
main_lr=0.01
n2v_lr=0.01 # 0.01
seed=0


experiment=linear_adv_test #sgd_final
# just retraining net2vecs to be more accurate
specific=bowling_alley
for seed in 1 2 3 4
do
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python3 post_train.py --seed 0 -leakage -nonlinear --model_extra _adv --n2v_extra _adv_after -with_n2v --experiment1 $experiment --experiment2 $experiment --module layer3 --ratio $ratio --n2v_lr 5e-5 --n2v_epochs $n2v_epochs --device $device2 --specific $specific 
    done
done

: '
specific=bowling_alley
for n in 15
do
    for ratio in 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.0 0.1 #0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.9 #0.0 1.0 #0.0 0.1 0.0 0.1 0.2 
    do
        python3 post_train.py --model_custom_end $n --n2v_custom_end $n -with_n2v --experiment1 sgd_final --experiment2 sgd_final --model_extra _debias --n2v_extra _debias_after --seed 0 -n2v --module layer3 --ratio $ratio --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device2 --specific $specific #-multiple
    done
done
'
