#!/bin/bash

device1=0
device2=0
device4=7
epochs=30 #10
main_lr=0.01
n2v_lr=0.1
module=layer3
#seed=0
specific=bowling_alley
alpha=1000
experiment=subset_sgd
: '
base_task(){
    wait $1
    python3 debias_train.py --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &
}

debias_task(){
    wait $1
    python3 debias_train.py -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module &
}

base_pid=""
debias_pid=""
# only train the resnets for now
for ratio in 0.1 0.2 #0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 
do
    # baseline
    base_task $base_pid &
    base_pid=$!
    # debias
    debias_task $debias_pid &
    debias_pid=$!
done
'

seed=1
specific=bowling_alley
# we are now training subsets...
for alpha in 1000
do
    for ratio in 0.6 0.7 0.8 0.9 1.0 0.0 0.2 0.3 0.4 0.5 0.1 
    do
        pids=""
        # baseline
    #    python3 debias_train.py --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &
    #    pids="$pids $!"
        # debias
        #python3 debias_train.py -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &

        # FINETUNED
        # BASELINES 
        python3 debias_train.py -subset --subset_ratio 0.025 --seed $seed -finetuned --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module --experiment $experiment 
      #  pids="$pids $!"
        #python3 debias_train.py --seed $trial -finetuned --alpha $alpha --ratio 0.0 --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device4 --specific $specific --module $module &
        #pids="$pids $!"

        # DEBIAS.....
        # adv
#        python3 debias_train.py -adv -nonlinear --seed $seed -finetuned --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.05 --epochs $epochs --device $device2 --specific $specific --module $module
#        python3 debias_train.py -adv --seed $seed -finetuned --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.01 --epochs $epochs --device $device2 --specific $specific --module $module

        #python3 debias_train.py -adv -nonlinear --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module

        python3 debias_train.py -subset --subset_ratio 0.025 --seed $seed -finetuned -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module --experiment $experiment
        #pids="$pids $!"

        #wait $pids
    done
done

# measuring TVAC on newly trained (not adversarially trained..)
: '
n2v_epochs=20
n2v_lr=0.01
seed=0
experiment=sgd_finetuned
model_extra=_adv
n2v_extra=_post_adv
# retrain net2vec?
for ratio in 0.1 0.9 1.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 #0.0 
do
    python3 post_train.py --experiment $experiment -with_n2v --model_extra $model_extra --n2v_extra $n2v_extra --seed $seed -n2v --module $module --ratio $ratio --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device4 --specific $specific
done
'
