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
metrics="accuracy tcav" #tcav # accuracy
: '
python3 compute_metrics.py -fresh -post --dataset coco --base_folder models/coco \
       --debias_extra 0_30_25 --module $module --epochs 30 --custom_end 25 \
       --metrics tcav --ratios 0 --seeds 0 \
       -debias --experiment sgd_final_noswitch_1e5_0.01 #--epochs None 10 20 30 40 -fresh

python3 compute_metrics.py -fresh -post --dataset coco --base_folder models/coco \
       --debias_extra 0 --module $module --epochs 30 --custom_end 30 \
       --metrics accuracy --ratios 0 --seeds 0 \
       -debias --experiment sgd_final_noswitch_1e5_0.01 #--epochs None 10 20 30 40 -fresh
'
#'
: '
for k in 5 10 15 20 25 30 35 40 45
do
    python3 compute_metrics.py -fresh -post --dataset coco --base_folder models/coco \
       --debias_extra 0_30_$k --module $module --epochs 30 --custom_end $k \
       --metrics tcav --ratios 0 --seeds 0 \
       -debias --experiment sgd_final_noswitch_1e5_0.01 #--epochs None 10 20 30 40 -fresh
done
'
: '
python3 compute_metrics.py -post --dataset coco --base_folder models/coco \
    --debias_extra 1 --module $module \
    --metrics $metrics --ratios 1 --seeds 0 --epochs 30 \
    -debias --experiment sgd_final_noswitch_jump2_3e4_0.01 #--epochs 10 20 30 40 -fresh
'
# baseline
#: '
python3 compute_metrics.py -fresh -post --dataset coco --base_folder models/coco \
    --module $module \
    --metrics tcav --ratios 1 --seeds 0 \
    -baseline --experiment sgd_finetuned
#'
#: '
# adversarial
: '
python3 compute_metrics.py -fresh -post --dataset coco --base_folder models/coco \
    --adv_extra 0 --module $module \
    --metrics accuracy --ratios 0 --seeds 0 \
    -adv --experiment sgd_finetuned
'
seed=0
#'
#gcloud config set compute/zone us-west1-b
# gcloud compute instances stop pytorch-1-vm 

