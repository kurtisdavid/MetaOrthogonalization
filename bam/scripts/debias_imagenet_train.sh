#!/bin/bash

device1=3
device2=7
device3=7
device4=4
epochs=30 #30 #10
main_lr=0.01
n2v_lr=0.1
module=layer3
#seed=0
specific=bowling_alley
alpha=1000

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

seed=0
specific=bowling_alley
#main_lr=0.001
# only train the resnets for now
for seed in 2
do
    for ratio in 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 0.0 0.1 0.2
    do
        pids=""
        # baseline
    #    python3 debias_train.py --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &
    #    pids="$pids $!"
        # debias
        #python3 debias_train.py -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &

        # FINETUNED
        # BASELINES 
       # python3 debias_train.py --seed $seed -finetuned --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module --experiment sgd_n2v_momentum --n2v_momentum 0.9 & 
        #python3 debias_train.py --seed $trial -finetuned --alpha $alpha --ratio 0.0 --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device4 --specific $specific --module $module &
#        pids="$pids $!"

        # DEBIAS.....
        # adv
#        python3 debias_train.py -adv --beta 1e-3 -nonlinear --seed $seed -finetuned --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment nonlinear_adv_test & 
 #       pids="$pids $!"
#        python3 debias_train.py -adv --beta 1e-5 --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device3 --specific $specific --module $module --experiment linear_adv_test
        
        #python3 debias_train.py -adv -nonlinear --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module
        python3 debias_train.py --seed $seed -finetuned --alpha 1000 --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 20 --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_imagenet_adaptive_resize --n2v_momentum 0.0 -imagenet -adaptive_resize -save_every -debias #-no_class #--train_bs 10

        #MULTIPLE
        # (BASELINE VERSION)
        #python3 debias_train.py --seed $seed -finetuned --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs 10 --device $device2 --specific $specific --module $module --experiment sgd_multiple -save_every -multiple
        #(debias version)
#        python3 debias_train.py --seed $seed -finetuned -debias --alpha 50 --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 20 --device $device2 --specific $specific --module $module --experiment1 sgd_multiple --experiment2 sgd_multiple_mu_generalized -save_every -n2v_start -multiple --n2v_momentum 0.9

        #python3 debias_train.py --seed $seed -finetuned -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs 15 --device $device2 --specific $specific --module $module --experiment sgd_reset3 -save_every -reset --reset_counter 3

        #python3 debias_train.py --seed $seed -finetuned -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module --experiment sgd_main_momentum --n2v_momentum 0.9 &
        #pids=5s $!"


        # LDA TRAINING!!
#        python3 debias_train.py -experimental --seed $seed -finetuned -debias --alpha .0001 --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs 10 --device $device2 --specific $specific --module $module --experiment sgd_lda  
        #pids="$pids $!"
        #wait $pids
    done
done
: '
# post train right after
for ratio in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do
    python3 post_train.py -with_n2v --experiment1 linear_adv_test_finetuned --experiment2 linear_adv_test_finetuned --model_extra _adv --n2v_extra _adv_after --seed 3 -n2v --module layer3 --ratio $ratio --n2v_lr 0.01 --n2v_epochs 20 --device $device3 --specific $specific
done
'

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
