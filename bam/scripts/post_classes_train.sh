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

# If you want to do 1 Class, 3 Class, etc, uncomment which # you want for specifics
specifics=("bowling_alley")
#specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope")
#specifics=("bowling_alley" "runway" "bamboo_forest" "bedroom" "corn_field" "track" "ski_slope" "cockpit" "bus_interior" "laundromat")
# specifics=("bowling_alley" "runway" "bamboo_forest")
# specifics=("bamboo_forest") 
declare specific;
printf -v specific "%s "  "${specifics[@]}"
# train both
for seed in 0 1 2 #0 3 4
do
    for ratio in 0.0 0.25 0.5 0.75 1.0 ##0.25 0.75 # 
    do
        pids=""
        
        python3 post_train.py --module layer3 --seed 0 -main -n2v --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device $device1 --specific $specific --experiment1 post_train --experiment2 post_train &
        pids="$pids $!"
        
        sleep 5
        python3 post_train.py --module layer3 --seed 1 -main -n2v --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device $device2 --specific $specific --experiment1 post_train --experiment2 post_train &
        pids="$pids $!"
        
        sleep 5
        python3 post_train.py --module layer3 --seed 2 -main -n2v --ratio $ratio --main_lr $main_lr --n2v_lr $n2v_lr --main_epochs $main_epochs --n2v_epochs $n2v_epochs --device $device3 --specific $specific --experiment1 post_train --experiment2 post_train &
        pids="$pids $!"
        wait $pids
        
    done
done

