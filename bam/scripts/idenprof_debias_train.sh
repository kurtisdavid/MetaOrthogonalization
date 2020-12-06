#!/bin/bash

device=2
device1=0
device2=1
device3=7
device4=6
epochs=15 #10 #15 #7 #20 #10 #30 #10
main_lr=0.001
n2v_lr=0.1
module=layer2
#seed=0
alpha=1000

big_alpha=1e3
n2v_lr=0.05
for alpha in 10 50 100 #5e3 1e4 1e5 # # #1 2 #1 2 #3 4
do
    for module in layer2 layer3 #0.25 0.5 0.75 1.0 #0.1 0.25 0.5 0.75 0.9 1.0 #0.0 0.5 1.0 #0.25 0.5 1.0 # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.2 0.4 0.6 0.8 1.0 #
    do
        pids=""
        # : '
        # FINETUNED, use these 3 on whatever settings to get baseline, adversarial, ours
        # python3 debias_train.py --dataset idenprof --base_folder models/idenprof --results_folder results/idenprof \
        #     --train_bs 64 --eval_bs 64 --seed $seed -finetuned --main_lr 0.001 --n2v_lr $n2v_lr \
        #     --epochs $epochs --device $device1 --module $module --experiment1 sgd --experiment2 sgd_final & #-save_every & #& #-save_every #-single
        # pids="$pids $!"
        # pids="$pids $!"
        # sleep 5
        # '

        # : '
        # MAKE SURE THAT NET2VECS ARE TRAINED PROPERLY
        device=1
        device1=4
        for seed in 0 1 2
        do
            python3 debias_train.py --dataset idenprof --base_folder models/idenprof --results_folder results/idenprof \
                --train_bs 64 --eval_bs 64 --seed $seed -finetuned -adv --beta $alpha --main_lr 0.0001 --n2v_lr .01 \
                --epochs $epochs  --device $device --module $module --experiment1 sgd --experiment2 linear_adv_test_stronger_$alpha & #-save_every & #& #-save_every #-single
            pids="$pids $!"
            sleep 5
            # python3 debias_train.py -partial_projection --dataset idenprof --base_folder models/idenprof --results_folder results/idenprof --train_bs 64 --eval_bs 64 \
            # -no_limit --gamma 0.0 --seed $seed -finetuned -debias --alpha $alpha --main_lr 0.0001 --n2v_lr .01 \
            # --epochs $epochs  --device $device1 --module $module --experiment1 sgd --experiment2 sgd_final_$alpha -constant_alpha & #-save_every & #& #-save_every #-single
            # pids="$pids $!"
            # sleep 5
            device=$((device + 1))
            device1=$((device1 + 1))
            # done
            pids="$pids $!"
        done
        wait $pids

        # python3 compute_metrics.py -fresh -post -debias --experiment sgd_final \
        # --device $device --module $module --metrics accuracy --ratios 0.5 \
        # --seeds $seeds --base_folder models/idenprof --dataset idenprof --debias_extra $alpha

        python3 compute_metrics.py -fresh -post -adv --experiment linear_adv_test_stronger_$alpha \
        --device $device3 --module $module --metrics $metrics --ratios 0.5 \
        --seeds $seeds --base_folder models/idenprof --dataset idenprof --adv_extra stronger_$alpha

        # wait $pids
    done
done

seeds="0 1 2"
ratios="0.5"
metrics="accuracy projection tcav leakage" #projection leakage tcav" #leakage" #"accuracy projection tcav"

# python3 compute_metrics.py -fresh -post -baseline --experiment sgd_final \
#         --device $device1 --module $module --metrics $metrics --ratios $ratios \
#         --seeds $seeds --base_folder models/idenprof --dataset idenprof &

# python3 compute_metrics.py -fresh -post -debias --experiment sgd_final_0.05 \
#         --device $device2 --module $module --metrics $metrics --ratios $ratios \
#         --seeds $seeds --base_folder models/idenprof --dataset idenprof &

# python3 compute_metrics.py -fresh -post -adv --experiment linear_adv_test \
#         --device $device3 --module $module --metrics $metrics --ratios $ratios \
#         --seeds $seeds --base_folder models/idenprof --dataset idenprof &