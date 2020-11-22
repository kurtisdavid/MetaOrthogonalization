#!/bin/bash
gcloud config set compute/zone us-west1-b
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
main_lr2=0.001
#--model_custom_end 9 --n2v_custom_end 9
experiment=sgd_final_noswitch_jump2_3e4_0.01
device=0
ratio=1
# baseline projection retraining
: '
python3 post_train.py --dataset coco --base_folder models/coco \
     -with_n2v --experiment1 sgd_finetuned --experiment2 sgd_finetuned \
     --model_extra _base --n2v_extra _base_after \
     -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 30 \
     --device 0 --ratio 0 --test_bs 32 --train_bs 100
'
#: '
python3 post_train.py --dataset coco --base_folder models/coco \
     -with_n2v --experiment1 sgd_finetuned --experiment2 sgd_finetuned \
     --model_extra _base --n2v_extra _base_after \
     -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 30 \
     --device 0 --ratio 1 --test_bs 64 --train_bs 100 -balanced
#'
# python3 post_train.py --dataset coco --base_folder models/coco -gender_balanced \
#     -with_n2v --experiment1 sgd_finetuned --experiment2 sgd_finetuned \
#     --model_extra _base --n2v_extra _base_after \
#     -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 25 \
#     --device 0 --ratio 0 --test_bs 64 --train_bs 100

# adversarial projection retraining
: '
python3 post_train.py --dataset coco --base_folder models/coco \
    -with_n2v --experiment1 sgd_finetuned --experiment2 sgd_finetuned \
    --model_extra _adv --n2v_extra _adv_after \
    -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 20  \
    --device 0 --ratio 0 --test_bs 32  --train_bs 64
'
# adversarial leakage?
: '
experiment=sgd_finetuned
for n2v_lr in 5e-5 0.001 0.01
do
    python3 post_train.py --dataset coco --base_folder models/coco -gender_balanced \
        -with_n2v --experiment1 $experiment --experiment2 $experiment \
        --model_extra _adv --n2v_extra _adv_after \
        -leakage -nonlinear --module layer3 --n2v_lr $n2v_lr --n2v_epochs 100 \
        --device 0 --ratio 0 --test_bs 64 --train_bs 64
done
'
: '
experiment=sgd_final_noswitch_1e5_0.01
for end in 10 20 30 40 #10 20 30 40 #2 #3 4  #0.05
do
    python3 post_train.py --dataset coco --base_folder models/coco \
      -with_n2v --experiment1 $experiment --experiment2 $experiment \
      --model_extra _debias --n2v_extra _debias_after --model_custom_end $end --n2v_custom_end $end \
      -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 25 \
      --device 0 --ratio 0 --test_bs 32 --train_bs 100 #-balanced
done
'
experiment=sgd_final_noswitch_1e5_0.01
: '
python3 post_train.py --dataset coco --base_folder models/coco \
      -with_n2v --experiment1 $experiment --experiment2 $experiment \
      --model_extra _debias --n2v_extra _debias_after --model_custom_end 30 --n2v_custom_end 30 \
      -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 30 \
      --device 0 --ratio 0 --test_bs 32 --train_bs 100
'
: '
python3 post_train.py --dataset coco --base_folder models/coco \
    -with_n2v --experiment1 sgd_finetuned --experiment2 sgd_finetuned \
    --model_extra _adv --n2v_extra _adv_after \
    -n2v --module layer3 --n2v_lr 0.001 --n2v_epochs 32 \
    --device 0 --ratio 0 --test_bs 64  --train_bs 64

'
: '
experiment=sgd_final_noswitch_jump2_3e4_0.01
for end in 30 #2 #3 4  #0.05
do
    python3 post_train.py --dataset coco --base_folder models/coco \
      -with_n2v --experiment1 $experiment --experiment2 $experiment \
      --model_extra _debias --n2v_extra _debias_after --model_custom_end $end --n2v_custom_end $end \
      -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 30 \
      --device 0 --ratio 1 --test_bs 32 --train_bs 100 -balanced
done
'
experiment=sgd_final_noswitch_jump2_3e4_0.01
: '
python3 post_train.py --dataset coco --base_folder models/coco \
      -with_n2v --experiment1 $experiment --experiment2 $experiment \
      --model_extra _debias --n2v_extra _debias_after \
      -n2v --module layer3 --n2v_lr 0.01 --n2v_epochs 15 \
      --device 0 --ratio 1 --test_bs 32 --train_bs 100 -balanced
'
: '
experiment=sgd_final_noswitch_1e5_0.01
ratio=0
# NOW COMPUTE LEAKAGE
for n2v_lr in 0.001 0.01 5e-5 #0.001 0.01
do
    python3 post_train.py --dataset coco --base_folder models/coco -gender_balanced \
        -with_n2v --experiment1 $experiment --experiment2 $experiment \
        --model_extra _debias --n2v_extra _debias_after --model_custom_end 30 --n2v_custom_end 30 \
        -leakage -nonlinear --module layer3 --n2v_lr $n2v_lr --n2v_epochs 100 \
        --device 0 --ratio $ratio --test_bs 128 --train_bs 128
done
'
#experiment=sgd_final_noswitch_jump_1e4_0.01
ratio=1
# NOW COMPUTE LEAKAGE
: '
n2v_lr=0.001
for end in 20 40 10 30 #5e-5 0.001 0.01
do
    python3 post_train.py --dataset coco --base_folder models/coco -gender_balanced \
        -with_n2v --experiment1 $experiment --experiment2 $experiment \
        --model_extra _debias --n2v_extra _debias_after --model_custom_end $end --n2v_custom_end $end \
        -leakage -nonlinear --module layer3 --n2v_lr $n2v_lr --n2v_epochs 100 \
        --device 0 --ratio $ratio -balanced --test_bs 128 --train_bs 128
done
'

: '
experiment=sgd_finetuned
n2v_lr=0.01
for n2v_lr in 0.001 0.01
do
     #python3 post_train.py --dataset coco --base_folder models/coco -gender_balanced \
#	-with_n2v --experiment1 $experiment --experiment2 $experiment \
#	--model_extra _base --n2v_extra _base_after \
#	-leakage -nonlinear --module layer3 --n2v_lr $n2v_lr --n2v_epochs 100 \
#	--device 0 --ratio 1 -balanced --test_bs 128 --train_bs 128
    python3 post_train.py --dataset coco --base_folder models/coco -gender_balanced \
        -with_n2v --experiment1 $experiment --experiment2 $experiment \
        --model_extra _base --n2v_extra _base_after \
        -leakage -nonlinear --module layer3 --n2v_lr $n2v_lr --n2v_epochs 100 \
        --device 0 --ratio 0 --test_bs 128 --train_bs 128
done
'

# modules=( "layer3" "layer4" "layer2" )
# for module in "${modules[@]}"
# do
#     for ratio in 0
#     do
#         python3 post_train.py --dataset coco --base_folder models/coco --seed 0 \
#             -main --main_lr $main_lr2 --main_epochs $main_epochs \
#         #     -n2v --n2v_lr $n2v_lr --n2v_epochs $n2v_epochs --module $module \
#             --experiment1 post_train --experiment2 post_train \
#             --test_bs 32 --train_bs 32 \
#             --ratio $ratio --device 7 #& #-parallel --gpu_ids 5 6 & #-balanced
#     done
# done
# python3 post_train.py --dataset coco --base_folder models/coco --module layer3 \
#         --seed 0 -main --main_lr $main_lr3 --main_epochs $main_epochs \
#         --experiment1 post_train --experiment2 post_train \
#         --test_bs 32 --train_bs 128 \
#         -balanced --ratio 2 -parallel --device 2 --gpu_ids 3 4 #&
# python3 post_train.py --dataset coco --base_folder models/coco --module layer3 \
#         --seed 0 -main --main_lr $main_lr3 --main_epochs $main_epochs \
#         --experiment1 post_train --experiment2 post_train \
#         --test_bs 32 --train_bs 128 \
#         -balanced --ratio 1 -parallel --device 7 --gpu_ids 5 6 #&
pids="$pids $!"

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
#gcloud compute instances stop pytorch-1-vm 
