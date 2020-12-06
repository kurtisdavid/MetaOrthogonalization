#!/bin/bash

device=3
device1=0
device2=2 #7
device3=4
device4=0
main_epochs=30 #30
n2v_epochs=20 #20
main_lr=0.01
n2v_lr=0.01 # 0.01
seed=0
: '
# only train the resnets for now
for ratio in 0.0 # 0.1 0.2 0.3 0.4 0.5 # 0.6 0.7 0.8 0.9 1.0 
do
    python3 post_train.py --seed 0 -main --ratio $ratio --main_lr $main_lr --main_epochs $main_epochs --device $device1 --specific bowling_alley &
#    python3 post_train.py --seed 2 -main --ratio $ratio --main_lr $main_lr --main_epochs $main_epochs --device $device2 --specific bowling_alley
done
'

# idenprof
main_lr1=0.01
main_lr2=0.001
main_lr3=0.0001
main_epochs=20

modules=( "conv1" "layer1" "layer2" "layer3" "layer4" "layer1.0" "layer1.1" "layer1.2" )
modules1=( "layer2.0" "layer2.1" "layer2.2" "layer2.3" "layer3.0" "layer3.1" "layer3.2" "layer3.3" )
modules2=( "layer3.4" "layer3.5" "layer4.0" "layer4.1" "layer4.2" )


: '
pids=""
python3 post_train.py --dataset idenprof --base_folder models/idenprof --module layer3 \
        --seed 0 -main --main_lr $main_lr2 --main_epochs $main_epochs \
        --device 0 --experiment1 post_train --experiment2 post_train \
        --test_bs 32 &
pids="$pids $!"
sleep 2
python3 post_train.py --dataset idenprof --base_folder models/idenprof --module layer3 \
        --seed 1 -main --main_lr $main_lr2 --main_epochs $main_epochs \
        --device 1 --experiment1 post_train --experiment2 post_train \
        --test_bs 32 &
pids="$pids $!"
sleep 2
python3 post_train.py --dataset idenprof --base_folder models/idenprof --module layer3 \
        --seed 2 -main --main_lr $main_lr2 --main_epochs $main_epochs \
        --device 2 --experiment1 post_train --experiment2 post_train \
        --test_bs 32 &
pids="$pids $!"
sleep 2
wait $pids
'
# module=layer2
# device=1
# pids=""
# for seed in 0 1 2
# do
#     python3 post_train.py --dataset idenprof --base_folder models/idenprof --module $module \
#         --seed $seed -n2v --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs \
#         --device $device --experiment1 post_train --experiment2 post_train \
#         --test_bs 32 &
#     pids="$pids $!"
#     sleep 2
#     device=$((device + 1))
# done
# wait $pids
: '
device=0
pids=""
for module in "${modules[@]}"
do
    # both always commented for now
    # python3 post_train.py --module layer3 --seed $seed -main -n2v --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device $device1 --specific $specific --experiment1 post_train --experiment2 post_train &
    python3 post_train.py --dataset idenprof --base_folder models/idenprof --module $module \
        --seed 0 -n2v --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs \
        --device $device --experiment1 post_train --experiment2 post_train \
        --test_bs 32 &
    pids="$pids $!"
    sleep 2
    
    device=$((device + 1))
done
wait $pids

device=0
pids=""
for module in "${modules1[@]}"
do
    # both always commented for now
    # python3 post_train.py --module layer3 --seed $seed -main -n2v --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device $device1 --specific $specific --experiment1 post_train --experiment2 post_train &
    python3 post_train.py --dataset idenprof --base_folder models/idenprof --module $module \
        --seed 0 -n2v --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs \
        --device $device --experiment1 post_train --experiment2 post_train \
        --test_bs 32 &
    pids="$pids $!"
    sleep 2
    
    device=$((device + 1))
done
wait $pids

device=0
pids=""
for module in "${modules2[@]}"
do
    # both always commented for now
    # python3 post_train.py --module layer3 --seed $seed -main -n2v --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device $device1 --specific $specific --experiment1 post_train --experiment2 post_train &
    python3 post_train.py --dataset idenprof --base_folder models/idenprof --module $module \
        --seed 0 -n2v --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs \
        --device $device --experiment1 post_train --experiment2 post_train \
        --test_bs 32 &
    pids="$pids $!"
    sleep 2
    
    device=$((device + 1))
done
wait $pids
'

: '
# only do if you want non linear versions
        python3 post_train.py --module layer3 --seed 0 -nonlinear -n2v --ratio 0.0 --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device 0 --specific $specific --experiment1 post_train --experiment2 post_train &
        sleep 5
        python3 post_train.py --module layer3 --seed 0 -nonlinear -n2v --ratio 0.25 --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device 1 --specific $specific --experiment1 post_train --experiment2 post_train &
        sleep 5
        python3 post_train.py --module layer3 --seed 0 -nonlinear -n2v --ratio 0.5 --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device 3 --specific $specific --experiment1 post_train --experiment2 post_train &
        sleep 5
        python3 post_train.py --module layer3 --seed 0 -nonlinear -n2v --ratio 0.75 --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device 4 --specific $specific --experiment1 post_train --experiment2 post_train &
        sleep 5
        python3 post_train.py --module layer3 --seed 0 -nonlinear -n2v --ratio 1.0 --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device 5 --specific $specific --experiment1 post_train --experiment2 post_train &
'



: '
# train net2vecs only
for seed in 0 1 2 3 4
do
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 # 
    do
        python3 post_train.py --module layer3 --seed $seed -n2v --ratio $ratio --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device2 --specific $specific --experiment1 post_train --experiment2 post_train
    done
done
'

: '
#module=layer2.1
specific=bowling_alley
# train net2vec only, but it is now an mlp ( when nonlinear on )
for ratio in 0.2 0.0 0.9 1.0 0.1 0.3 0.4 0.5 0.6 0.7 0.8
do
    python3 post_train.py --seed 1 -n2v --module layer3 --ratio $ratio --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device1 --specific $specific &
    python3 post_train.py --seed 2 -n2v --module layer3 --ratio $ratio --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device2 --specific $specific 
done
'
: '
specific=bowling_alley
python3 post_train.py --seed $seed -n2v --module layer3.0 --ratio 0.0 --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device2 --specific $specific
python3 post_train.py --seed $seed -n2v --module layer3.3 --ratio 0.7 --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device2 --specific $specific
'


: '
# just retraining net2vecs to be more accurate
specific=bowling_alley
for experiment in sgd_final #linear_adv_test #sgd_30_finetuned #subset_sgd_finetuned 
do
    for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    do
        python3 post_train.py -with_n2v --experiment1 $experiment --experiment2 $experiment --model_extra _adv --n2v_extra _adv_after --seed 0 -n2v --module layer3 --ratio $ratio --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --device $device2 --specific $specific 
    done
done
'
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
