#!/bin/bash

device=2
device1=2
device2=6
device3=7
device4=6
epochs=20 #15 #10 #15 #7 #20 #10 #30 #10
main_lr=0.001
n2v_lr=0.1
module=layer3
#seed=0
alpha=1000

seed=0
gcloud config set compute/zone us-west1-b
for ratio in 1
do
    ct=0
    for n2v_lr in 0.01 #0.05
    do
        for big_alpha in 3e4 # 1e6 # 3e3
        do
            exp2=sgd_final_noswitch_jump2_
            exp2+=$big_alpha
            exp2+="_"
            exp2+=$n2v_lr
            pids=""
            # FINETUNED, use these 3 on whatever settings to get baseline, adversarial, ours
            # baseline
	    : '
            python3 debias_train.py --seed $seed -finetuned --module $module \
                 --dataset coco --base_folder models/coco --results_folder results/coco \
                 --main_lr $main_lr --n2v_lr $n2v_lr --epochs 50 \
                 --experiment1 sgd --experiment2 sgd_finetuned \
                 --train_bs 32 --test_bs 64 --ratio 0 \
                 --alpha $big_alpha -constant_alpha \
                 --device 0 -probe_eval_off & #-mean_debias -no_limit & #-debias -balanced
                # --device $(( 0 + $ratio )) &
	    '
            # python3 debias_train.py --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final &
            pids="$pids $!"
            sleep 5 
	    : '
            python3 -W ignore debias_train.py --seed $seed -finetuned --module $module \
                --dataset coco --base_folder models/coco --results_folder results/coco \
                --main_lr $main_lr --n2v_lr $n2v_lr --epochs 50 \
                --experiment1 sgd --experiment2 $exp2 \
                --train_bs 32 --test_bs 32 --ratio $ratio \
                --alpha $big_alpha -save_every -balanced \
                --device 0 -debias -probe_eval_off -no_limit -jump_alpha & #-debias & #-mean_debias -no_limit & #-debias -balanced -constant_alpha 
                # --device $(( 0 + $ratio )) &
	    '
            # python3 debias_train.py --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final &
            pids="$pids $!"
            sleep 5
            ct=$(( 1 + $ct ))
            # '

            # : '
            # MAKE SURE THAT NET2VECS ARE TRAINED PROPERLY
            # python3 debias_train.py -adv --beta 0.5 --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 linear_adv_test_stronger &
            # pids="$pids $!"
            # sleep 5
            # '
            
            wait $pids
        done
    done
done

