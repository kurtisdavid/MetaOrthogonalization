#!/bin/bash

device=6
device1=5
device2=6
device3=2
device4=7
epochs=7 #10 #30 #10
main_lr=0.01
n2v_lr=0.1
module=layer3
#seed=0
specific=bowling_alley
alpha=1000

specific=bowling_alley #runway
# only train the resnets for now
for alpha in 250 500 #750 1000 1250 #100 250 500 #1200 1300 1400 1500 # runway #bowling_alley # runway 
    do
    for seed in 0 # 0 2 3 4
    do
        for ratio in 0.0 0.25 0.5 0.75 1.0 #0.0 0.25 0.5 0.75 1.0 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.2 0.4 0.6 0.8 1.0 #
        do
            pids=""
            # baseline
        #    python3 debias_train.py --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &
        #    pids="$pids $!"
            # debias
            #python3 debias_train.py -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &

            # FINETUNED, use these 3 on whatever settings to get baseline, adversarial, ours
            
#            python3 debias_train.py --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device1 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final & 
#            pids="$pids $!"
#            sleep 5
            # MAKE SURE THAT NET2VECS ARE TRAINED PROPERLY
            # python3 debias_train.py -adv --beta 5e-2 --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device4 --specific $specific --module $module --experiment1 sgd --experiment2 linear_adv_test &
            # pids="$pids $!"
            # sleep 5
            #python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 1000 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final & #-save_every #-single
            #pids="$pids $!"
            #sleep 5
            

            # non linear adv
    #        python3 debias_train.py -adv --beta 1e-3 -nonlinear --seed $seed -finetuned --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment nonlinear_adv_test & 
     #       pids="$pids $!"
            # both?
            #python3 debias_train.py -adv -nonlinear --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module
            
            #CURRENT SETTING FOR OURS
            # MOST CURRENT SETTING FOR OURS (currently folder is different, make sure to change back)
            # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 1000 --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 1 --device $device --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_neutral & #-save_every #-single
            # pids="$pids $!"
            # sleep 5
            
            # testing GAMMA
            python3 debias_train.py --gamma 0.1 --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --combined_n2v_lr 0.01 --epochs 20 --device $device1 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_pos_0.1_$alpha -partial_projection --bias_norm l2 & #-single -save_every
            pids="$pids $!"
            sleep 5
            python3 debias_train.py --gamma -0.1 --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --combined_n2v_lr 0.01 --epochs 20 --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_neg_0.1_$alpha -partial_projection --bias_norm l2 & #-single -save_every
            pids="$pids $!"
            sleep 5
            # python3 debias_train.py --gamma 0.5 --seed $seed -finetuned -debias --alpha 10 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 16 --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_pos_0.5 -save_every & #-single
            # pids="$pids $!"
            # sleep 5
            # python3 debias_train.py --gamma 1.0 --seed $seed -finetuned -debias --alpha 10 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 16 --device $device4 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_pos_1.0 -save_every & #-single
            # pids="$pids $!"
            # sleep 5
            
            # baseline
            #python3 debias_train.py --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 7 --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final #-save_every #-single


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
            wait $pids
        done
    done
done

for alpha in 250 500 #750 1000 1250 #100 250 500 #1200 1300 1400 1500 # runway #bowling_alley # runway 
    do
    for seed in 0 # 0 2 3 4
    do
        for ratio in 0.0 0.25 0.5 0.75 1.0 #0.0 0.25 0.5 0.75 1.0 #0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.2 0.4 0.6 0.8 1.0 #
        do
            pids=""
            # baseline
        #    python3 debias_train.py --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &
        #    pids="$pids $!"
            # debias
            #python3 debias_train.py -debias --alpha $alpha --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device1 --specific $specific --module $module &

            # FINETUNED, use these 3 on whatever settings to get baseline, adversarial, ours
            
#            python3 debias_train.py --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device1 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final & 
#            pids="$pids $!"
#            sleep 5
            # MAKE SURE THAT NET2VECS ARE TRAINED PROPERLY
            # python3 debias_train.py -adv --beta 5e-2 --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device4 --specific $specific --module $module --experiment1 sgd --experiment2 linear_adv_test &
            # pids="$pids $!"
            # sleep 5
            #python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 1000 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final & #-save_every #-single
            #pids="$pids $!"
            #sleep 5
            

            # non linear adv
    #        python3 debias_train.py -adv --beta 1e-3 -nonlinear --seed $seed -finetuned --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment nonlinear_adv_test & 
     #       pids="$pids $!"
            # both?
            #python3 debias_train.py -adv -nonlinear --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --epochs $epochs --device $device2 --specific $specific --module $module
            
            #CURRENT SETTING FOR OURS
            # MOST CURRENT SETTING FOR OURS (currently folder is different, make sure to change back)
            # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 1000 --seed $seed --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 1 --device $device --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_neutral & #-save_every #-single
            # pids="$pids $!"
            # sleep 5
            
            # testing GAMMA
            python3 debias_train.py --gamma 0.1 --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --combined_n2v_lr 0.01 --epochs 20 --device $device1 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_pos_0.1_L1_$alpha -partial_projection --bias_norm l1 & #-single -save_every
            pids="$pids $!"
            sleep 5
            python3 debias_train.py --gamma -0.1 --seed $seed -finetuned -debias --alpha $alpha --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --combined_n2v_lr 0.01 --epochs 20 --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_neg_0.1_L1_$alpha -partial_projection --bias_norm l1 & #-single -save_every
            pids="$pids $!"
            sleep 5
            # python3 debias_train.py --gamma 0.5 --seed $seed -finetuned -debias --alpha 10 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 16 --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_pos_0.5 -save_every & #-single
            # pids="$pids $!"
            # sleep 5
            # python3 debias_train.py --gamma 1.0 --seed $seed -finetuned -debias --alpha 10 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 16 --device $device4 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_gamma_pos_1.0 -save_every & #-single
            # pids="$pids $!"
            # sleep 5
            
            # baseline
            #python3 debias_train.py --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 7 --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final #-save_every #-single


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
            wait $pids
        done
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
