#!/bin/bash

device=2
device1=2
device2=6
device3=7
device4=6
epochs=15 #10 #15 #7 #20 #10 #30 #10
main_lr=0.001
n2v_lr=0.1
module=layer3
#seed=0
alpha=1000

# specifics=("bowling_alley" "runway" "bamboo_forest")
specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope")
# specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field")
#specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope" "cockpit" "bus_interior" "laundromat")
# specifics=("bowling_alley")
# specifics=("bamboo_forest")
declare specific;
printf -v specific "%s "  "${specifics[@]}"

big_alpha=1500
n2v_lr=0.05
for seed in 1 2 #1 2 #3 4
do
    for ratio in 1.0 #0.25 0.5 0.75 1.0 #0.1 0.25 0.5 0.75 0.9 1.0 #0.0 0.5 1.0 #0.25 0.5 1.0 # 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 #0.2 0.4 0.6 0.8 1.0 #
    do
        pids=""
        : '
        # FINETUNED, use these 3 on whatever settings to get baseline, adversarial, ours
        python3 debias_train.py --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final &
        pids="$pids $!"
        sleep 5
        '

        : '
        # MAKE SURE THAT NET2VECS ARE TRAINED PROPERLY
        python3 debias_train.py -adv --beta 0.5 --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 linear_adv_test_stronger &
        pids="$pids $!"
        # sleep 5
        '
        
        # debias
        python3 debias_train.py -no_limit --gamma 0.0 --seed $seed -finetuned -debias --alpha $big_alpha --ratio $ratio --main_lr .001 --n2v_lr $n2v_lr --epochs $epochs  --device 0 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final2_0.05 -constant_alpha # -constant_alpha -save_every & #& #-save_every #-single
        pids="$pids $!"
        #sleep 5
        # '

        # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 50 --ratio $ratio --main_lr $main_lr --n2v_lr 0.01 --epochs 19 --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_50_1000 -jump_alpha -save_every & #-save_every & #& #-save_every #-single
        # pids="$pids $!"

        # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 50 --ratio $ratio --main_lr $main_lr --n2v_lr 0.01 --epochs 19 --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final -jump_alpha & #-save_every & #& #-save_every #-single
        # pids="$pids $!"
        
        # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 100 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 10 --device $device --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_100_jump -jump_alpha -save_every & #& #-save_every #-single
        # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 100 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs 19 --device $device --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_100_jump -linear_alpha & # -save_every & #& #-save_every #-single
        : '
        for seed in 0 #1 2
        do
            device=$(( 7 + $seed ))
            # good main_lr = .001 n2v_lr = 0.01
            python3 debias_train.py -no_limit --gamma 0.0 --seed $seed -finetuned -debias --alpha 1000 --ratio $ratio --main_lr .01 --n2v_lr 0.1 --epochs 10 --device $device --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_old_1000 -constant_alpha & #-save_every & #& #-save_every #-single
            # python3 debias_train.py -no_limit --gamma 0.0 --seed $seed -finetuned -debias --alpha 1000 --ratio $ratio --main_lr .001 --n2v_lr 0.01 --epochs 19 --device $device --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_100_1000 -jump_alpha & #-save_every & #& #-save_every #-single
            # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 50 --ratio $ratio --main_lr $main_lr --n2v_lr 0.01 --epochs 19 --device $device --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_50_1000 -jump_alpha -save_every & #-save_every & #& #-save_every #-single
            pids="$pids $!"
            sleep 5
        done
        '
        : '
        python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 250 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_250 -constant_alpha & #& #-save_every #-single
        pids="$pids $!"
        sleep 5
        '
        : '
        # WEAKER REGULARIZATION
        python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 1000 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device3 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_750 -constant_alpha & #& #-save_every #-single
        pids="$pids $!"
        '
        
        : '
        # nonlinear adversary
        python3 debias_train.py -nonlinear -adv --beta 5e-2 --seed $seed -finetuned --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device2 --specific $specific --module $module --experiment1 sgd --experiment2 nonlinear_adv_test &
        pids="$pids $!"
        sleep 5
        '
        # sleep 5
        # python3 debias_train.py --gamma 0.0 --seed $seed -finetuned -debias --alpha 1250 --ratio $ratio --main_lr $main_lr --n2v_lr 0.1 --epochs $epochs --device $device4 --specific $specific --module $module --experiment1 sgd --experiment2 sgd_final_1250 -constant_alpha & #& #-save_every #-single
        # pids="$pids $!"
        # sleep 5
        # pids="$pids $!"
        # sleep 5
        wait $pids
    done
done

# python3 compute_metrics.py -fresh -post -adv --experiment linear_adv_test_stronger --adv_extra stronger --device $device2 --specific $specific --module $module --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2

# og
# python3 compute_metrics.py -fresh -post -debias --experiment sgd_final_$big_alpha --debias_extra $big_alpha --device $device3 --specific $specific --module $module --metrics accuracy projection --ratios 0.0 --seeds 0
# python3 compute_metrics.py -final -post --metrics accuracy projection --debias_extra $big_alpha --specific $specific --module $module --ratios 0.0 --seeds 0



# specifics=("bowling_alley" "runway" "bamboo_forest")
# specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope")
# specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field")
# specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope" "cockpit" "bus_interior" "laundromat")
# specifics=("bowling_alley")
# specifics=("bamboo_forest")
# declare specific;
# printf -v specific "%s "  "${specifics[@]}"
: '
# # baseline
python3 compute_metrics.py -fresh -post -baseline --experiment sgd_final \
    --device $device2 --specific $specific --module $module --metrics accuracy projection tcav leakage \
    --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2 &
sleep 2
# # adv
python3 compute_metrics.py -fresh -post -adv --adv_extra stronger \
    --experiment linear_adv_test_stronger --device $device3 --specific $specific \
    --module $module --metrics accuracy projection tcav leakage --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2
'
# DEBIAS METRICS
# python3 compute_metrics.py -fresh -post -debias --experiment sgd_final_$n2v_lr --debias_extra $n2v_lr --device $device1 --specific $specific --module $module --metrics accuracy projection tcav leakage --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2
# python3 compute_metrics.py -final -post --debias_extra $n2v_lr --specific $specific --module $module --metrics accuracy projection tcav leakage --ratios 0.0 0.25 0.5 0.75 1.0 --seeds 0 1 2

# python3 compute_metrics.py -final -post --metrics accuracy projection --debias_extra $big_alpha --specific $specific --module $module --ratios 0.0 --seeds 0
