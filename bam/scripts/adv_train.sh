#!/bin/bash

device1=3
device2=3
device3=3
device4=7
epochs=30 #10
main_lr=0.01
n2v_lr=0.01
module=layer3
#module=fc
seed=0
specific=bowling_alley
alpha=100
beta=0.001

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

# only train the resnets for now
for ratio in 0.5 # 0.8 0.9 # 0.1 0.2 0.3 0.4 0.5 # 0.6 0.7 0.8 0.9 1.0 
do
    pids=""
    # baseline
#    python3 debias_train.py --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device4 --specific $specific --module $module &
#    pids="$pids $!"
    # debias
    #python3 debias_train.py -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &

    ### FINETUNED ###

    # baseline
#    python3 debias_train.py --seed $seed -finetuned --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device3 --specific $specific --module $module &
    pids="$pids $!"

    # debias
#    python3 debias_train.py --seed $seed -finetuned -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module &
#:    pids="$pids $!"

    # adv
    python3 debias_train.py --seed $seed -finetuned -adv --beta $beta --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module --experiment linear_adv_test
#    pids="$pids $!"


    wait $pids
done
